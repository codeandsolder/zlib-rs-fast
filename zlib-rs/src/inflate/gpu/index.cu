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
// Exact block-relative extents. Required history is validated when the
// real chain reaches a candidate.
struct BlockExtent {
    u64 header_bit,header_end_bit,end_bit;
    u32 length,required_history,kind,last,status,reserved;
};
static_assert(sizeof(BlockExtent)==48);

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
struct TileSelection {
    u32 entry,offset,count,end;
};
static_assert(sizeof(TileSelection)==16);
static_assert(sizeof(TileBlock)==24 && sizeof(TileTask)==8 && sizeof(TilePath)==16);
static_assert(sizeof(TileTrees)==2392);

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
        u32 position=threadIdx.x,length=0,need=0,status=0,tokens=0;
        if(position>=size)status=2;
        while(position<size){
            const u32 edge=edges[position],step=edge&63;
            if(!step){status=2;break;}
            const u32 bytes=(edge>>6)&511,distance=edge>>15;
            position+=step;
            if(!bytes){status=1;break;}
            if(distance>length)need=max(need,distance-length);
            length+=bytes;++tokens;
        }
        if(status==0 && size<tile_bits)status=2;
        paths[static_cast<u64>(blockIdx.x)*max_token_bits+threadIdx.x]={length,need,position,status|(tokens<<2)};
    }
}

// GPU index ABI. Segments have the same layout as the bounded decoder's
// descriptors; intermediate descriptors use block-relative output offsets.
constexpr u32 segment_bytes=128*1024;
struct IndexInfo {
    u32 status,count,decoded,window,adler,groups,tokens,reserved;
};
static_assert(sizeof(IndexInfo)==32);

__device__ bool emit_segment(Segment* out,u32 capacity,u32& count,
    u64 header,u64 header_end,u64 start_bit,u64 end_bit,
    u32 start,u32 end,u32 kind,bool last) {
    if(start==end)return true;
    if(count==capacity)return false;
    out[count++]={header,header_end,start_bit,end_bit,start,end,kind,last?1u:0u};
    return true;
}

__device__ bool emit_token_segment(Segment* dest,TokenRange* ranges,u32 capacity,u32& emitted,
    u64 header,u64 header_end,u64 start_bit,u64 end_bit,u32 start,u32 end,bool last,
    u32 first_token,u32 token_end) {
    const u32 slot=emitted;
    if(!emit_segment(dest,capacity,emitted,header,header_end,start_bit,end_bit,start,end,2,last))
        return false;
    if(emitted!=slot)ranges[slot]={first_token,token_end-first_token};
    return true;
}

// Produce independently indexed block-relative segments by composing only
// the token paths reachable from the real beginning of this candidate.
// Each tile consumes at most 559 bits (at least two bits per match), so its
// decoded span is strictly smaller than the 128 KiB segment bound.
extern "C" __global__ void inflate_compose_segments(const TileBlock* blocks,const TileTrees* trees,
    const TilePath* paths,u32 count,u32 limit,BlockExtent* extents,Segment* segments,u32* counts,TokenRange* ranges,TileSelection* selected) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const TileBlock b=blocks[i];
    const TileTrees& t=trees[i];
    BlockExtent out{b.bit,t.payload,0,0,0,2,t.last,1,0};
    u32 emitted=0;
    Segment* dest=segments+b.first_tile;
    TokenRange* token_ranges=ranges+b.first_tile;
    if(!t.status){
        u32 entry=0,seg_start=0,token_total=0,seg_token=0;
        u64 seg_bit=t.payload;
        for(u32 tile=0;tile<b.tile_count;++tile){
            const u64 start=t.payload+static_cast<u64>(tile)*tile_bits;
            if(start>=b.upper)break;
            const TilePath path=paths[(static_cast<u64>(b.first_tile)+tile)*max_token_bits+entry];
            const u32 status=path.status&3,tokens=path.status>>2;
            if(status==2 || path.length>limit-out.length || path.length>segment_bytes)break;
            if(path.required_history>out.length)
                out.required_history=max(out.required_history,path.required_history-out.length);
            if(out.length-seg_start+path.length>segment_bytes){
                if(!emit_token_segment(dest,token_ranges,b.tile_count,emitted,b.bit,t.payload,seg_bit,start+entry,
                    seg_start,out.length,false,seg_token,token_total))break;
                seg_start=out.length;seg_bit=start+entry;seg_token=token_total;
            }
            selected[b.first_tile+tile]={entry,token_total,tokens,path.step|(status<<16)};
            token_total+=tokens;
            out.length+=path.length;
            if(status==1){
                out.end_bit=start+path.step;
                if(emit_token_segment(dest,token_ranges,b.tile_count,emitted,b.bit,t.payload,seg_bit,out.end_bit,
                    seg_start,out.length,true,seg_token,token_total)){
                    out.status=0;out.reserved=token_total;
                }
                break;
            }
            if(path.step<tile_bits || path.step>=tile_bits+max_token_bits)break;
            entry=path.step-tile_bits;
        }
    }
    extents[i]=out;
    counts[i]=out.status?0:emitted;
}

// Exact fallback for fixed/stored blocks and dynamic candidates whose
// speculative interval contained a false header. No candidate assumption
// is used here. The absolute output position checks the history bound.
__device__ bool scan_emit(const u8* input,u64 end,u64& bit,u32& decoded,u32 window,u32 limit,
    Segment* segments,u32 capacity,u32& count,bool& last,u32& status,u32& serial_tokens) {
    const u64 header=bit;
    Reader r{input,0,bit,end};
    last=r.take(1)!=0;
    const u32 kind=r.take(2);
    status=1;
    if(!r.good||kind==3)return false;
    if(kind==0){
        r.position=(r.position+7)&~7ull;
        const u32 length=r.take(16),inverse=r.take(16);
        if(!r.good||(length^inverse)!=65535 || r.position+8ull*length>end)return false;
        if(length>limit-decoded){status=2;return false;}
        bit=r.position+8ull*length;
        if(!emit_segment(segments,capacity,count,header,r.position,r.position,bit,decoded,decoded+length,0,true)){
            status=3;return false;
        }
        decoded+=length;return true;
    }
    Tree ll,dd;
    if(!tables(r,kind,ll,dd))return false;
    const u64 header_end=r.position;
    u64 start_bit=header_end;
    u32 start=decoded;
    while(r.good){
        if(!serial_tokens){status=3;return false;}
        --serial_tokens;
        const u64 token_bit=r.position;
        const u32 symbol=ll.decode(r);
        if(!r.good)return false;
        if(symbol==256){
            bit=r.position;
            if(!emit_segment(segments,capacity,count,header,header_end,start_bit,bit,start,decoded,kind,true)){
                status=3;return false;
            }
            return true;
        }
        u32 length=1,distance=0;
        if(symbol>=257 && !match(r,dd,symbol,length,distance))return false;
        if(distance>min(decoded,window))return false;
        if(length>limit-decoded){status=2;return false;}
        if(decoded-start+length>segment_bytes){
            if(!emit_segment(segments,capacity,count,header,header_end,start_bit,token_bit,start,decoded,kind,false)){
                status=3;return false;
            }
            start=decoded;start_bit=token_bit;
        }
        decoded+=length;
    }
    return false;
}

struct Placement {
    u32 dest,decoded,count,token_base;
};
static_assert(sizeof(Placement)==16);
// Walk only proven edges from the zlib member's actual first bit. Dynamic
// candidate extents never authorize a jump unless their exact header offset
// is reached, their history is legal, and their decoded bound is satisfied.
// Fixed/stored gaps and failed speculative candidates are parsed exactly.
extern "C" __global__ void inflate_assemble_index(const u8* input,u64 size,
    const TileBlock* blocks,const BlockExtent* extents,
    const u32* counts,u32 candidates,u32 limit,Segment* segments,u32 capacity,Placement* placements,IndexInfo* info) {
    if(threadIdx.x||blockIdx.x)return;
    *info={1,0,0,0,0,0,0,0};
    if(size<6)return;
    const u32 cmf=input[0],flg=input[1];
    if((cmf&15)!=8 || (cmf>>4)>7 || ((cmf<<8)|flg)%31 || (flg&32))return;
    const u32 window=256u<<(cmf>>4);
    info->window=window;
    const u64 body_end=(size-4)*8;
    u64 bit=16;
    u32 decoded=0,count=0,candidate=0,serial_tokens=1u<<20,tokens=0;
    bool last=false;
    while(!last){
        if(bit>=body_end)return;
        while(candidate<candidates && blocks[candidate].bit<bit)++candidate;
        if(candidate<candidates && blocks[candidate].bit==bit && !extents[candidate].status){
            const BlockExtent extent=extents[candidate];
            if(extent.end_bit<=bit || extent.end_bit>body_end || extent.required_history>decoded)return;
            if(extent.length>limit-decoded){info->status=2;return;}
            const u32 n=counts[candidate];
            if(n>capacity-count){info->status=3;return;}
            placements[candidate]={count,decoded,n,tokens};
            count+=n;tokens+=extent.reserved;
            decoded+=extent.length;bit=extent.end_bit;last=extent.last!=0;
            ++candidate;
        }else{
            u32 status=1;
            if(!scan_emit(input,body_end,bit,decoded,window,limit,segments,capacity,count,last,status,serial_tokens)){
                info->status=status;return;
            }
        }
    }
    if((bit+7)/8!=size-4)return;
    const u32 adler=(static_cast<u32>(input[size-4])<<24)|(static_cast<u32>(input[size-3])<<16)|
        (static_cast<u32>(input[size-2])<<8)|input[size-1];
    *info={0,count,decoded,window,adler,0,tokens,0};
}

// Copy only the candidates reached by the exact chain. Destinations are
// disjoint, with serial fixed/stored fallback segments already written.
extern "C" __global__ void inflate_place_segments(const TileBlock* blocks,
    const Segment* relative,const Placement* placements,Segment* segments,const IndexInfo* info,
    const TokenRange* relative_ranges,TokenRange* ranges) {
    if(info->status)return;
    const Placement p=placements[blockIdx.x];
    const Segment* source=relative+blocks[blockIdx.x].first_tile;
    for(u32 i=threadIdx.x;i<p.count;i+=blockDim.x){
        Segment seg=source[i];seg.start+=p.decoded;seg.end+=p.decoded;
        segments[p.dest+i]=seg;
        TokenRange range=relative_ranges[blocks[blockIdx.x].first_tile+i];
        range.offset+=p.token_base;
        ranges[p.dest+i]=range;
    }
}

// Decode only the proven token path inside each tile, in parallel across
// the full member. The compact token list avoids serial Huffman parsing
// in the dictionary expansion stage. The same exact bit endpoint is
// checked again, including the terminal EOB.
extern "C" __global__ void inflate_emit_tokens(const u8* input,const TileBlock* blocks,
    const TileTrees* trees,const TileTask* tasks,const TileSelection* selected,
    const Placement* placements,u32 tiles,u32 capacity,u32* tokens,IndexInfo* info,u32 window) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tiles||info->status||info->tokens>capacity)return;
    const TileTask task=tasks[i];
    const Placement p=placements[task.block];
    const TileSelection select=selected[i];
    if(!p.count||!select.count)return;
    const TileBlock b=blocks[task.block];
    const TileTrees& t=trees[task.block];
    const u64 start=t.payload+static_cast<u64>(task.tile)*tile_bits;
    Reader r{input,0,start+select.entry,b.upper};
    if(p.token_base>info->tokens || select.offset>info->tokens-p.token_base ||
        select.count>info->tokens-p.token_base-select.offset){atomicOr(&info->status,4u);return;}
    for(u32 j=0;j<select.count;++j){
        const u32 symbol=t.ll.decode(r);
        u32 value=0;
        if(!r.good||symbol==256){atomicOr(&info->status,4u);return;}
        if(symbol<256)value=symbol;
        else {
            u32 length=0,distance=0;
            if(!match(r,t.dd,symbol,length,distance)||distance>window){atomicOr(&info->status,4u);return;}
            value=(length<<16)|distance;
        }
        tokens[p.token_base+select.offset+j]=value;
    }
    if((select.end>>16)==1 && t.ll.decode(r)!=256)r.good=false;
    if(!r.good||r.position!=start+(select.end&65535))atomicOr(&info->status,4u);
}

// Ordered compaction of validated dynamic headers. The bitmap is one bit
// per compressed bit; unlike an atomic append, compaction preserves order
// without a host sort. Each chunk covers 128 words (four warps).
constexpr u32 header_chunk_words=128;
extern "C" __global__ void inflate_validate_bitmap(const u8* input,u64 end,
    const u64* candidates,u32 count,u32* bitmap) {
    const u32 i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const u64 bit=candidates[i];
    Reader r{input,0,bit,end};
    r.take(1);
    if(r.take(2)!=2)return;
    Tree ll,dd;
    if(tables(r,2,ll,dd,false))atomicOr(bitmap+bit/32,1u<<(bit%32));
}
extern "C" __global__ void inflate_count_header_chunks(const u32* bitmap,u32 words,u32* offsets) {
    const u32 i=blockIdx.x*header_chunk_words+threadIdx.x;
    u32 count=i<words?__popc(bitmap[i]):0;
    for(u32 delta=16;delta;delta/=2)count+=__shfl_down_sync(0xffffffffu,count,delta);
    __shared__ u32 warp_counts[4];
    if(threadIdx.x%32==0)warp_counts[threadIdx.x/32]=count;
    __syncthreads();
    if(!threadIdx.x)offsets[blockIdx.x]=warp_counts[0]+warp_counts[1]+warp_counts[2]+warp_counts[3];
}
extern "C" __global__ void inflate_header_offsets(u32* offsets,u32 chunks) {
    if(threadIdx.x||blockIdx.x)return;
    u32 sum=0;
    for(u32 i=0;i<chunks;++i){const u32 n=offsets[i];offsets[i]=sum;sum+=n;}
    offsets[chunks]=sum;
}
extern "C" __global__ void inflate_scatter_headers(const u32* bitmap,u32 words,const u32* offsets,u64* headers) {
    const u32 i=blockIdx.x*header_chunk_words+threadIdx.x;
    u32 bits=i<words?bitmap[i]:0;
    const u32 count=__popc(bits),lane=threadIdx.x%32,warp=threadIdx.x/32;
    u32 prefix=count;
    for(u32 delta=1;delta<32;delta*=2){
        const u32 prev=__shfl_up_sync(0xffffffffu,prefix,delta);
        if(lane>=delta)prefix+=prev;
    }
    __shared__ u32 warp_counts[4];
    if(lane==31)warp_counts[warp]=prefix;
    __syncthreads();
    u32 slot=offsets[blockIdx.x]+prefix-count;
    for(u32 j=0;j<warp;++j)slot+=warp_counts[j];
    while(bits){
        const u32 bit=__ffs(bits)-1;
        headers[slot++]=static_cast<u64>(i)*32+bit;
        bits&=bits-1;
    }
}
extern "C" __global__ void inflate_make_tile_specs(const u64* headers,u32 count,u64 end,
    TileBlock* blocks,u32* total_tiles) {
    if(threadIdx.x||blockIdx.x)return;
    u32 first=0;
    for(u32 i=0;i<count;++i){
        const u64 bit=headers[i],upper=i+1<count?headers[i+1]:end;
        const u32 tiles=static_cast<u32>((upper-bit+tile_bits-1)/tile_bits);
        blocks[i]={bit,upper,first,tiles};first+=tiles;
    }
    *total_tiles=first;
}
extern "C" __global__ void inflate_make_tile_tasks(const TileBlock* blocks,TileTask* tasks) {
    const TileBlock b=blocks[blockIdx.x];
    for(u32 tile=threadIdx.x;tile<b.tile_count;tile+=blockDim.x)
        tasks[b.first_tile+tile]={blockIdx.x,tile};
}

struct DecodeGroup {
    u32 offset,count,start,end;
};
static_assert(sizeof(DecodeGroup)==16);
// The full segment list remains resident. Only these bounded group
// descriptors are needed by the host to launch the compression windows.
// Jump directly to the next needed window even if a segment crosses
// thousands of very small windows.
extern "C" __global__ void inflate_schedule_groups(const Segment* segments,IndexInfo* info,
    u32 compression_window,u32 max_group,DecodeGroup* groups) {
    if(threadIdx.x||blockIdx.x||info->status)return;
    if(!compression_window||!max_group){info->status=1;return;}
    u32 offset=0,position=0,group_count=0;
    while(offset<info->count){
        const u32 target=static_cast<u32>(min(static_cast<u64>(info->decoded),
            (static_cast<u64>(position)/compression_window+1)*compression_window));
        const u32 begin=offset,start=position;
        do {
            const Segment seg=segments[offset++];
            if(seg.start!=position||seg.end<=position||seg.end>info->decoded){
                info->status=1;return;
            }
            position=seg.end;
        }while(offset<info->count && offset-begin<max_group && position<target);
        groups[group_count++]={begin,offset-begin,start,position};
    }
    if(position!=info->decoded){info->status=1;return;}
    info->groups=group_count;
}
} // namespace resident_inflate
