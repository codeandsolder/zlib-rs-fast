namespace resident_inflate {
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
struct Segment { u64 header_bit,header_end_bit,start_bit,end_bit; u32 start,end,kind,ends_block; };
static_assert(sizeof(Segment)==48);
struct TokenRange {u32 offset,count;};
static_assert(sizeof(TokenRange)==8);
struct Reader {
    const u8* input;
    u64 origin, position, end;
    bool good=true;
    __device__ u32 peek(u32 n) {
        if(position>=end || position/8<origin) {good=false;return 0;}
        const u64 byte=position/8;
        u32 v=0;
        for(u32 i=0;i<4 && (byte+i)*8<end;++i) v|=static_cast<u32>(input[byte+i-origin])<<(8*i);
        return (v>>(position%8))&((1u<<n)-1);
    }
    __device__ void skip(u32 n) {
        if(position>end || n>end-position){good=false;return;}
        position+=n;
    }
    __device__ u32 take(u32 n) {
        if(n==0)return 0;
        if(position>end || n>end-position){good=false;return 0;}
        const u32 result=peek(n);position+=n;return result;
    }
};
constexpr u32 max_bits=15;
constexpr u32 lookup_bits=8;
constexpr u32 lookup_size=1u<<lookup_bits;
struct Tree {
    u16 symbols[288];
    u16 lookup[lookup_size];
    u16 counts[16], first[16], offsets[16];
    u32 width;
    __device__ bool build(const u8* lengths,u32 n,bool allow_empty=false,bool allow_single=true,bool materialize=true) {
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
        if(code+counts[width]!=(1u<<width) && !(allow_single&&width==1&&offset==1))return false;
        if(!materialize)return true;
        u16 next[16];
        for(u32 i=0;i<=width;++i)next[i]=offsets[i];
        for(u32 i=0;i<n;++i)if(lengths[i])symbols[next[lengths[i]]++]=static_cast<u16>(i);
        // Store (symbol << 4) | code length. Zero marks a long or absent
        // code. Canonical prefixes are contiguous in reversed bit order.
        for(u32 i=0;i<lookup_size;++i)lookup[i]=0;
        for(u32 bits=1;bits<=min(width,lookup_bits);++bits){
            for(u32 j=0;j<counts[bits];++j){
                const u32 start=(static_cast<u32>(first[bits])+j)<<(lookup_bits-bits);
                const u16 entry=static_cast<u16>((symbols[offsets[bits]+j]<<4)|bits);
                for(u32 k=0;k<(1u<<(lookup_bits-bits));++k)lookup[start+k]=entry;
            }
        }
        return true;
    }
    __device__ u32 decode(Reader& r) const {
        if(!width){r.good=false;return 0;}
        // Reverse the lookahead once, then compare canonical prefixes.
        const u32 bits=__brev(r.peek(max(width,lookup_bits)));
        const u16 entry=lookup[bits>>(32-lookup_bits)];
        if(entry){
            r.skip(entry&15);return entry>>4;
        }
        for(u32 n=lookup_bits+1;n<=width;++n){
            const u32 code=bits>>(32-n);
            if(code>=first[n] && code-first[n]<counts[n]){
                r.skip(n);return symbols[offsets[n]+code-first[n]];
            }
        }
        r.good=false;return 0;
    }
};
__device__ __constant__ u8 cl_order[19]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
__device__ bool tables(Reader& r,u32 type,Tree& ll,Tree& dd,bool materialize=true) {
    u8 lengths[320]={};
    if(type==1){
        for(u32 i=0;i<288;++i)lengths[i]=i<144?8:i<256?9:i<280?7:8;
        if(!ll.build(lengths,288,false,true,materialize))return false;
        for(u32 i=0;i<32;++i)lengths[i]=5;
        return dd.build(lengths,32,false,true,materialize);
    }
    const u32 nl=r.take(5)+257,nd=r.take(5)+1,nc=r.take(4)+4;
    if(nl>286 || nd>30 || nc>19)return false;
    u8 cl[19]={};
    for(u32 i=0;i<nc;++i)cl[cl_order[i]]=static_cast<u8>(r.take(3));
    Tree ct;
    if(!r.good || !ct.build(cl,19,false,false))return false;
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
    return r.good && lengths[256] && ll.build(lengths,nl,false,true,materialize) && dd.build(lengths+nl,nd,true,true,materialize);
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
extern "C" __global__ void inflate_symbolic(const u8* input,const u8* first_header,
    u64 input_origin,u64 header_origin,u64 first_header_bit,const Segment* segments,u32 count,
    const u32* tokens,const TokenRange* ranges,u32 total_tokens,
    u16* symbolic,u32* status,u32* used_seeds,u32 window,u32 base) {
    const u32 index=(blockIdx.x*blockDim.x+threadIdx.x)/warp_width,lane=threadIdx.x%warp_width;
    if(index>=count)return;
    const Segment seg=segments[index];
    const bool first=seg.header_bit==first_header_bit;
    Reader header{first?first_header:input,first?header_origin:input_origin,seg.header_bit,seg.header_end_bit};
    Reader r{input,input_origin,seg.start_bit,seg.end_bit};
    Tree ll,dd;
    const TokenRange range=total_tokens?ranges[index]:TokenRange{0,0};
    const bool predecoded=range.count!=0;
    u32 ok=1,stored=0;
    if(!lane){
        if(predecoded){
            ok=seg.kind==2 && range.offset<=total_tokens && range.count<=total_tokens-range.offset;
        }else{
        header.take(1);const u32 type=header.take(2);
        ok=header.good && type==seg.kind && type!=3;
        if(ok && type==0){
            header.position=(header.position+7)&~7ull;
            stored=header.take(16);const u32 inverse=header.take(16);
            ok=header.good && (stored^inverse)==65535u && stored==seg.end-seg.start &&
                header.position==seg.start_bit && header.position+8ull*stored==seg.end_bit;
        }else if(ok){
            ok=tables(header,type,ll,dd) && header.position==seg.header_end_bit && header.position<=seg.start_bit;
        }
        }
    }
    ok=__shfl_sync(0xffffffffu,ok,0);
    stored=__shfl_sync(0xffffffffu,stored,0);
    if(!ok){if(!lane)status[index]=1;return;}
    if(seg.kind==0){
        const u64 start=seg.start_bit/8;
        for(u32 i=lane;i<stored;i+=warp_width)symbolic[seg.start-base+i]=input[start+i-input_origin];
        if(!lane)status[index]=0;return;
    }
    u32 pos=seg.start,token_index=0;
    for(;;){
        u32 symbol=0,length=0,distance=0;
        if(!lane){
            if(predecoded){
                if(token_index==range.count){symbol=256;ok=pos==seg.end;}
                else {
                    const u32 token=tokens[range.offset+token_index++];
                    length=token>>16;distance=token&65535;
                    if(!length){symbol=distance;length=1;distance=0;ok=symbol<256&&pos<seg.end;}
                    else {symbol=257;ok=length>=3&&length<=258&&distance&&distance<=min(pos,window)&&length<=seg.end-pos;}
                }
            }else if(!seg.ends_block && pos==seg.end){
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
        if(symbol<256){if(!lane)symbolic[pos-base]=static_cast<u16>(symbol);}
        else {
            // Self-overlap maps to the initial distance bytes, all of
            // which precede the token. Earlier segments are represented
            // symbolically, never read before their producer completes.
            for(u32 i=lane;i<length;i+=warp_width){
                const u32 source=pos-distance+(i%distance);
                if(source<seg.start){
                    const u32 seed=seg.start-source-1;
                    symbolic[pos+i-base]=static_cast<u16>(256+seed);
                    atomicOr(used_seeds+static_cast<u64>(index)*(window/32)+seed/32,1u<<(seed%32));
                }else symbolic[pos+i-base]=symbolic[source-base];
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
// Each cross-segment edge strictly decreases the segment index. Earlier
// decoded batches already have concrete bytes in the retained raw prefix.
// Resolve the seed only once; propagation inside a segment does not create
// any new seed IDs, so unused dictionary entries need never be resolved.
__device__ u8 resolve_seed(const Segment* segments,const u16* symbolic,const u8* raw,
    u32 index,u32 distance,u32 base,u32 raw_origin,u32* error) {
    u32 origin=segments[index].start,block=index;
    for(;;){
        if(distance>origin){atomicOr(error,4u);return 0;}
        const u32 source=origin-distance;
        if(source<base){
            if(source<raw_origin){atomicOr(error,4u);return 0;}
            return raw[source-raw_origin];
        }
        if(!block){atomicOr(error,4u);return 0;}
        u32 lo=0,hi=block;
        while(lo+1<hi){
            const u32 middle=lo+(hi-lo)/2;
            if(segments[middle].start<=source)lo=middle;else hi=middle;
        }
        block=lo;origin=segments[block].start;
        const u16 value=symbolic[source-base];
        if(value<256)return static_cast<u8>(value);
        distance=static_cast<u32>(value)-255;
    }
}
extern "C" __global__ void inflate_seed(const Segment* segments,const u16* symbolic,const u8* raw,
    const u32* used_seeds,u8* seeds,u32* error,u32 window,u32 base,u32 raw_origin) {
    if(*error)return;
    const u32 index=blockIdx.y,available=min(segments[index].start,window);
    const u32* used=used_seeds+static_cast<u64>(index)*(window/32);
    for(u32 word=blockIdx.x*blockDim.x+threadIdx.x;word<(available+31)/32;word+=gridDim.x*blockDim.x){
        u32 bits=used[word];
        while(bits){
            const u32 bit=__ffs(bits)-1,seed=word*32+bit;
            if(seed>=available){atomicOr(error,4u);return;}
            seeds[static_cast<u64>(index)*window+seed]=resolve_seed(segments,symbolic,raw,index,seed+1,base,raw_origin,error);
            bits&=bits-1;
        }
    }
}
extern "C" __global__ void inflate_resolve(const Segment* segments,const u16* symbolic,const u8* seeds,
    u8* raw,const u32* error,u32 window,u32 base,u32 raw_origin) {
    if(*error)return;
    const u32 index=blockIdx.y,start=segments[index].start,end=segments[index].end;
    const u8* dictionary=seeds+static_cast<u64>(index)*window;
    for(u32 pos=start+blockIdx.x*blockDim.x+threadIdx.x;pos<end;pos+=gridDim.x*blockDim.x){
        const u16 value=symbolic[pos-base];
        raw[pos-raw_origin]=value<256?static_cast<u8>(value):dictionary[value-256];
    }
}

} // namespace resident_inflate
