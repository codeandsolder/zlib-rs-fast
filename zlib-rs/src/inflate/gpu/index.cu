// Experimental speculative index discovery for ordinary DEFLATE streams.
// A dynamic header is only a candidate until an exact parse starting from
// the member's known first block reaches it. Never trust speculation alone.
namespace resident_inflate {
__device__ u32 index_word(const u32* data,u64 bit) {
    const u64 word=bit/32;
    const u32 shift=bit%32;
    return __funnelshift_r(data[word],data[word+1],shift);
}
// Packed source has at least three zero-padded words beyond its logical end.
extern "C" __global__ void inflate_header_candidates(const u32* input,u64 end,
    u64* candidates,u32 capacity,u32* count) {
    const u64 i=16+static_cast<u64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i+29>end)return;
    const u32 prefix=index_word(input,i);
    if((prefix&6)!=4 || ((prefix>>3)&31)>29 || ((prefix>>8)&31)>29)return;
    const u32 n=4+((prefix>>13)&15);
    if(i+17+3*n>end)return;
    // A valid code-length alphabet is a complete binary prefix tree.
    // The sum of weights 2^(7-length) must be exactly 128.
    u32 sum=0, bits=index_word(input,i+17);
    for(u32 j=0;j<n;++j){
        if(j==10)bits=index_word(input,i+47);
        const u32 length=bits&7;bits>>=3;
        if(length)sum+=128u>>length;
        if(sum>128)return;
    }
    if(sum!=128)return;
    const u32 slot=atomicAdd(count,1u);
    if(slot<capacity)candidates[slot]=i;
}
// Fully validate candidate code trees before attempting any token walk.
extern "C" __global__ void inflate_validate_headers(const u8* input,u64 end,
    const u64* candidates,u32 count,u64* validated,u32 capacity,u32* valid_count) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const u64 bit=candidates[i];
    Reader r{input,0,bit,end};
    r.take(1);
    if(r.take(2)!=2)return;
    Tree ll,dd;
    if(!tables(r,2,ll,dd))return;
    const u32 slot=atomicAdd(valid_count,1u);
    if(slot<capacity)validated[slot]=bit;
}
// Count tokens without materializing output. A valid candidate describes a
// block-relative result; required_history is the smallest amount of decoded
// history that makes every distance legal. A caller must check that when
// assembling the real chain. Length and bit bounds contain false positives.
struct BlockExtent {
    u64 header_bit,header_end_bit,end_bit;
    u32 length,required_history,kind,last,status,reserved;
};
static_assert(sizeof(BlockExtent)==48);
__device__ BlockExtent scan_extent(const u8* input,u64 end,u64 bit,u32 limit,u32 window) {
    BlockExtent out{bit,0,0,0,0,0,0,1,0};
    Reader r{input,0,bit,end};
    out.last=r.take(1);out.kind=r.take(2);
    if(!r.good || out.kind==3)return out;
    if(out.kind==0){
        r.position=(r.position+7)&~7ull;
        const u32 length=r.take(16),inverse=r.take(16);
        if(!r.good || (length^inverse)!=65535 || length>limit)return out;
        out.header_end_bit=r.position;
        if(r.position+8ull*length>end)return out;
        out.length=length;out.end_bit=r.position+8ull*length;out.status=0;
        return out;
    }
    Tree ll,dd;
    if(!tables(r,out.kind,ll,dd))return out;
    out.header_end_bit=r.position;
    while(r.good){
        const u32 symbol=ll.decode(r);
        if(!r.good)return out;
        if(symbol==256){out.end_bit=r.position;out.status=0;return out;}
        u32 length=1,distance=0;
        if(symbol>=257 && !match(r,dd,symbol,length,distance))return out;
        if(length>limit-out.length || distance>window)return out;
        if(distance>out.length)out.required_history=max(out.required_history,distance-out.length);
        out.length+=length;
    }
    return out;
}
extern "C" __global__ void inflate_candidate_extents(const u8* input,u64 end,
    const u64* candidates,u32 count,u32 limit,u32 window,BlockExtent* extents) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)extents[i]=scan_extent(input,end,candidates[i],limit,window);
}
} // namespace resident_inflate

namespace resident_inflate {
// Speculative tiles are bounded by the next plausible header. Such a
// header may occur inside a real entropy block: reaching the bound without
// EOB is not success. The exact scanner can then extend the interval.
constexpr u32 tile_bits=512;
constexpr u32 max_token_bits=48;
struct TileBlock {
    u64 bit,upper;
    u32 first_tile,tile_count;
};
struct TileTask { u32 block,tile; };
struct TileTrees {
    Tree ll,dd;
    u64 payload;
    u32 last,status;
};
struct TilePath {
    u32 length,required_history,step,status;
};
static_assert(sizeof(TileBlock)==24 && sizeof(TileTask)==8 && sizeof(TilePath)==16);

extern "C" __global__ void inflate_prepare_tiles(const u8* input,const TileBlock* blocks,
    u32 count,TileTrees* trees) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const TileBlock b=blocks[i];
    TileTrees& t=trees[i];
    Reader r{input,0,b.bit,b.upper};
    t.last=r.take(1);
    t.status=1;
    if(r.take(2)!=2 || !tables(r,2,t.ll,t.dd))return;
    t.payload=r.position;t.status=0;
}
__device__ u32 token_edge(const Tree& ll,const Tree& dd,Reader r,u32 window) {
    const u64 begin=r.position;
    const u32 symbol=ll.decode(r);
    if(!r.good)return 0;
    u32 length=1,distance=0;
    if(symbol==256)length=0;
    else if(symbol>=257 && !match(r,dd,symbol,length,distance))return 0;
    const u32 step=static_cast<u32>(r.position-begin);
    if(!step || step>max_token_bits || distance>window)return 0;
    // 6 bit step, 9 bit output length, 16 bit distance.
    return step|(length<<6)|(distance<<15);
}
extern "C" __global__ void inflate_tile_paths(const u8* input,const TileBlock* blocks,
    const TileTrees* trees,const TileTask* tasks,u32 window,TilePath* paths) {
    const TileTask task=tasks[blockIdx.x];
    const TileBlock b=blocks[task.block];
    const TileTrees& t=trees[task.block];
    if(t.status)return;
    const u64 start=t.payload+static_cast<u64>(task.tile)*tile_bits;
    if(start>=b.upper)return;
    const u32 size=static_cast<u32>(min(static_cast<u64>(tile_bits),b.upper-start));
    __shared__ Tree ll,dd;
    __shared__ u32 edges[tile_bits];
    // Copy each validated table once into shared memory. It is read-only
    // during the independent token decodes.
    for(u32 j=threadIdx.x;j<sizeof(Tree)/sizeof(u32);j+=blockDim.x){
        reinterpret_cast<u32*>(&ll)[j]=reinterpret_cast<const u32*>(&t.ll)[j];
        reinterpret_cast<u32*>(&dd)[j]=reinterpret_cast<const u32*>(&t.dd)[j];
    }
    __syncthreads();
    for(u32 j=threadIdx.x;j<size;j+=blockDim.x){
        edges[j]=token_edge(ll,dd,Reader{input,0,start+j,b.upper},window);
    }
    __syncthreads();
    if(threadIdx.x<max_token_bits){
        u32 position=threadIdx.x,length=0,need=0,status=0;
        if(position>=size)status=2;
        while(position<size){
            const u32 edge=edges[position],step=edge&63;
            if(!step){status=2;break;}
            const u32 bytes=(edge>>6)&511,distance=edge>>15;
            position+=step;
            if(!bytes){status=1;break;}
            if(distance>length)need=max(need,distance-length);
            length+=bytes;
        }
        if(status==0 && size<tile_bits)status=2;
        paths[static_cast<u64>(blockIdx.x)*max_token_bits+threadIdx.x]={length,need,position,status};
    }
}
extern "C" __global__ void inflate_compose_tiles(const TileBlock* blocks,const TileTrees* trees,
    const TilePath* paths,u32 count,u32 limit,BlockExtent* extents) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const TileBlock b=blocks[i];
    const TileTrees& t=trees[i];
    BlockExtent out{b.bit,t.payload,0,0,0,2,t.last,1,0};
    if(!t.status){
        u32 entry=0;
        for(u32 tile=0;tile<b.tile_count;++tile){
            const u64 start=t.payload+static_cast<u64>(tile)*tile_bits;
            if(start>=b.upper)break;
            const TilePath path=paths[(static_cast<u64>(b.first_tile)+tile)*max_token_bits+entry];
            if(path.status==2 || path.length>limit-out.length)break;
            if(path.required_history>out.length)
                out.required_history=max(out.required_history,path.required_history-out.length);
            out.length+=path.length;
            if(path.status==1){out.end_bit=start+path.step;out.status=0;break;}
            if(path.step<tile_bits || path.step>=tile_bits+max_token_bits)break;
            entry=path.step-tile_bits;
        }
    }
    extents[i]=out;
}
} // namespace resident_inflate
