// Experimental block-indexed CUDA DEFLATE decoder.
// Index entries are bit offsets in the zlib member and logical output
// offsets. Every block is checked against both of its indexed endpoints.
// Symbols below 256 are known bytes; other symbols refer to a byte in the
// 32 KiB history preceding the segment (distance = symbol - 255).
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
constexpr u32 history_limit=32768;
constexpr u32 warp_width=32;
struct Segment { u64 header_bit,start_bit,end_bit; u32 start,end,kind,ends_block; };
static_assert(sizeof(Segment)==40);
struct Reader {
    const u8* input;
    u64 position, end;
    bool good=true;
    __device__ u32 peek(u32 n) {
        if(position>=end) {good=false;return 0;}
        const u64 byte=position/8;
        u32 v=0;
        for(u32 i=0;i<4 && (byte+i)*8<end;++i) v|=static_cast<u32>(input[byte+i])<<(8*i);
        return (v>>(position%8))&((1u<<n)-1);
    }
    __device__ u32 take(u32 n) {
        if(n==0)return 0;
        if(position>end || n>end-position){good=false;return 0;}
        const u32 result=peek(n);position+=n;return result;
    }
};
constexpr u32 max_bits=15;
struct Tree {
    u16 symbols[288];
    u16 counts[16], first[16], offsets[16];
    u32 width=0;
    __device__ bool build(const u8* lengths,u32 n,bool allow_empty=false) {
        for(u32 i=0;i<=max_bits;++i)counts[i]=first[i]=offsets[i]=0;
        width=0;
        for(u32 i=0;i<n;++i){
            if(lengths[i]>max_bits)return false;
            if(lengths[i]){++counts[lengths[i]];width=max(width,static_cast<u32>(lengths[i]));}
        }
        if(!width)return allow_empty;
        u32 code=0,offset=0;
        for(u32 i=1;i<=width;++i){
            code=(code+counts[i-1])*2;
            if(code+counts[i]>(1u<<i))return false;
            first[i]=static_cast<u16>(code);
            offsets[i]=static_cast<u16>(offset);
            offset+=counts[i];
        }
        if(code+counts[width]!=(1u<<width) && !(width==1&&offset==1))return false;
        u16 next[16];
        for(u32 i=0;i<=width;++i)next[i]=offsets[i];
        for(u32 i=0;i<n;++i)if(lengths[i])symbols[next[lengths[i]]++]=static_cast<u16>(i);
        return true;
    }
    __device__ u32 decode(Reader& r) const {
        if(!width){r.good=false;return 0;}
        // Reverse the lookahead once, then compare canonical prefixes.
        const u32 bits=__brev(r.peek(width));
        for(u32 n=1;n<=width;++n){
            const u32 code=bits>>(32-n);
            if(code>=first[n] && code-first[n]<counts[n]){
                r.take(n);return symbols[offsets[n]+code-first[n]];
            }
        }
        r.good=false;return 0;
    }
};
__device__ __constant__ u8 cl_order[19]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
__device__ bool tables(Reader& r,u32 type,Tree& ll,Tree& dd) {
    u8 lengths[320]={};
    if(type==1){
        for(u32 i=0;i<288;++i)lengths[i]=i<144?8:i<256?9:i<280?7:8;
        if(!ll.build(lengths,288))return false;
        for(u32 i=0;i<32;++i)lengths[i]=5;
        return dd.build(lengths,32);
    }
    const u32 nl=r.take(5)+257,nd=r.take(5)+1,nc=r.take(4)+4;
    if(nl>286 || nd>32 || nc>19)return false;
    u8 cl[19]={};
    for(u32 i=0;i<nc;++i)cl[cl_order[i]]=static_cast<u8>(r.take(3));
    Tree ct;
    if(!r.good || !ct.build(cl,19))return false;
    for(u32 i=0;i<nl+nd && r.good;){
        const u32 s=ct.decode(r);
        if(s<16)lengths[i++]=static_cast<u8>(s);
        else {
            if(s>18 || (s==16&&!i))return false;
            const u32 n=s==16?r.take(2)+3:s==17?r.take(3)+3:r.take(7)+11;
            if(n>nl+nd-i)return false;
            const u8 value=s==16?lengths[i-1]:0;
            for(u32 j=0;j<n;++j)lengths[i++]=value;
        }
    }
    return r.good && lengths[256] && ll.build(lengths,nl) && dd.build(lengths+nl,nd,true);
}
__device__ bool match(Reader& r,const Tree& dd,u32 code,u32& length,u32& dist) {
    if(code<257||code>285)return false;
    if(code<=264)length=code-254;
    else if(code==285)length=258;
    else {
        const u32 x=code-265,extra=x/4+1;
        length=3+((4+x%4)<<extra)+r.take(extra);
    }
    const u32 d=dd.decode(r);
    if(d>=30)return false;
    if(d<4)dist=d+1;
    else {
        const u32 extra=d/2-1;
        dist=1+((2+d%2)<<extra)+r.take(extra);
    }
    return r.good;
}
// A segment starts and ends at exact token boundaries, possibly inside
// a large entropy block. Rebuild the same tree from its original header.
// One warp decodes independently, using symbolic preceding history.
extern "C" __global__ void inflate_symbolic(const u8* input,const Segment* segments,u32 count,
    u16* symbolic,u32* status,u32 window) {
    const u32 index=(blockIdx.x*blockDim.x+threadIdx.x)/warp_width,lane=threadIdx.x%warp_width;
    if(index>=count)return;
    const Segment seg=segments[index];
    Reader r{input,seg.header_bit,seg.end_bit};
    Tree ll,dd;
    u32 ok=1,stored=0;
    if(!lane){
        r.take(1);const u32 type=r.take(2);
        ok=r.good && type==seg.kind && type!=3;
        if(ok && type==0){
            r.position=(r.position+7)&~7ull;
            stored=r.take(16);const u32 inverse=r.take(16);
            ok=r.good && (stored^inverse)==65535u && stored==seg.end-seg.start &&
                r.position==seg.start_bit && r.position+8ull*stored==seg.end_bit;
        }else if(ok){
            ok=tables(r,type,ll,dd) && r.position<=seg.start_bit;
            r.position=seg.start_bit;
        }
    }
    ok=__shfl_sync(0xffffffffu,ok,0);
    stored=__shfl_sync(0xffffffffu,stored,0);
    if(!ok){if(!lane)status[index]=1;return;}
    if(seg.kind==0){
        const u64 start=seg.start_bit/8;
        for(u32 i=lane;i<stored;i+=warp_width)symbolic[seg.start+i]=input[start+i];
        if(!lane)status[index]=0;return;
    }
    u32 pos=seg.start;
    for(;;){
        u32 symbol=0,length=0,distance=0;
        if(!lane){
            if(!seg.ends_block && pos==seg.end){
                symbol=256;ok=r.position==seg.end_bit;
            } else {
                symbol=ll.decode(r);
                if(!r.good)ok=0;
                else if(symbol<256){length=1;ok=pos<seg.end;}
                else if(symbol==256){ok=seg.ends_block && pos==seg.end&&r.position==seg.end_bit;}
                else ok=match(r,dd,symbol,length,distance)&&distance<=min(pos,window)&&length<=seg.end-pos;
            }
        }
        ok=__shfl_sync(0xffffffffu,ok,0);
        symbol=__shfl_sync(0xffffffffu,symbol,0);
        length=__shfl_sync(0xffffffffu,length,0);
        distance=__shfl_sync(0xffffffffu,distance,0);
        if(!ok||symbol==256)break;
        if(symbol<256){if(!lane)symbolic[pos]=static_cast<u16>(symbol);}
        else {
            // Self-overlap maps to the initial distance bytes, all of
            // which precede the token. Earlier segments are represented
            // symbolically, never read before their producer completes.
            for(u32 i=lane;i<length;i+=warp_width){
                const u32 source=pos-distance+(i%distance);
                symbolic[pos+i]=source<seg.start?static_cast<u16>(255+seg.start-source):symbolic[source];
            }
        }
        __syncwarp();
        pos+=length;
    }
    if(!lane)status[index]=ok?0:2;
}
// Refuse to dereference any symbolic output if a segment failed validation.
extern "C" __global__ void inflate_status(const u32* status,u32 count,u32* error) {
    for(u32 i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        if(status[i])atomicOr(error,status[i]);
}
// Resolve symbolic references, which strictly decrease the originating
// segment. Binary search only when crossing a segment boundary.
extern "C" __global__ void inflate_resolve(const Segment* segments,u32 count,const u16* symbolic,
    u8* output,u32* error) {
    if(*error)return;
    const u32 index=blockIdx.y;
    const u32 start=segments[index].start,end=segments[index].end;
    for(u32 pos=start+blockIdx.x*blockDim.x+threadIdx.x;pos<end;pos+=gridDim.x*blockDim.x){
        u32 origin=start,block=index;
        u16 value=symbolic[pos];
        while(value>=256){
            const u32 distance=static_cast<u32>(value)-255;
            if(distance>origin||!block){atomicOr(error,4u);return;}
            const u32 source=origin-distance;
            u32 lo=0,hi=block;
            while(lo+1<hi){
                const u32 middle=lo+(hi-lo)/2;
                if(segments[middle].start<=source)lo=middle;else hi=middle;
            }
            block=lo;origin=segments[block].start;value=symbolic[source];
        }
        output[pos]=static_cast<u8>(value);
    }
}
