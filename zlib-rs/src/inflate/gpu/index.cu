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
