// GPU-resident independent DEFLATE blocks, sharing the real sliding history.
// The host submits one window (history + core), and only reads compressed
// bytes and the combined Adler components. All intermediate data stays here.
using u16 = unsigned short;
using u64 = unsigned long long;
constexpr u32 hash_count = 65536;
constexpr u32 adler_mod = 65521;
constexpr u32 warp_size = 32;
struct Tile {
    u32 start, origin, len, core;
};
__device__ __forceinline__ Tile tile_at(u32 tile, u32 size, u32 history, u32 tile_bytes, u32 max_dist) {
    const u32 start = history + tile * tile_bytes;
    const u32 origin = start > max_dist-1 ? start-(max_dist-1) : 0;
    const u32 end = start+min(tile_bytes,size-start);
    return {start, origin, end-origin, end-start};
}
__device__ __forceinline__ Tile search_tile_at(u32 tile,u32 size,u32 history,u32 tile_bytes,u32 max_dist) {
    Tile t=tile_at(tile,size,history,tile_bytes,max_dist);
    t.len+=min(match_limit-1,size-t.start-t.core);
    return t;
}
__device__ __forceinline__ u32 hash_at(const u32* input, u32 pos) {
    return ((input[pos]&0xffffffu)*2654435761u)>>16;
}
// One warp per tile. Same-hash lanes find their nearest earlier lane by
// ballot, and only the last lane updates the hash head. All head loads are
// ordered before writes; processing 32 positions at a time preserves the
// exact serial nearest-predecessor relation, including hash collisions.
extern "C" __global__ void resident_index(const u32* input, u32* heads_all, u16* predecessors,
    u32 size, u32 history, u32 tile_bytes, u32 stride, u32 max_dist, u32 tiles) {
    const u32 lane=threadIdx.x%warp_size;
    const u32 tile=(blockIdx.x*blockDim.x+threadIdx.x)/warp_size;
    if (tile>=tiles) return;
    const Tile t=search_tile_at(tile,size,history,tile_bytes,max_dist);
    u32* heads=heads_all+static_cast<u64>(tile)*hash_count;
    u16* pred_out=predecessors+static_cast<u64>(tile)*stride;
    for (u32 h=lane; h<hash_count; h+=warp_size) heads[h]=0;
    __syncwarp();
    for (u32 base=0; base<t.len; base+=warp_size) {
        const u32 pos=base+lane;
        const u32 active=__ballot_sync(0xffffffffu,pos+2<t.len);
        if (pos+2<t.len) {
            const u32 hash=hash_at(input,t.origin+pos);
            const u32 peers=__match_any_sync(active,hash);
            const u32 lower=peers&((1u<<lane)-1);
            const u32 previous=lower ? base+31-__clz(lower)+1 : heads[hash];
            const u32 distance=previous ? pos+1-previous : 0;
            pred_out[pos]=static_cast<u16>(distance<max_dist ? distance : 0);
            __syncwarp(active);
            if (lane==31-__clz(peers)) heads[hash]=pos+1;
        } else if (pos<t.len) {
            pred_out[pos]=0;
        }
        __syncwarp();
    }
}
// Only search core positions. History is present in the index and input,
// but we never compute match results for the unused history halo.
extern "C" __global__ void resident_match(const u32* input,const u32* predecessors,u32* matches,
    u32 size,u32 history,u32 tile_bytes,u32 stride,u32 max_dist,u32 budget) {
    const u32 tile=blockIdx.y;
    const Tile t=search_tile_at(tile,size,history,tile_bytes,max_dist);
    const u32* words=input+t.origin;
    const u32* p=predecessors+static_cast<u64>(tile)*(stride/2);
    u32* output=matches+static_cast<u64>(tile)*tile_bytes;
    const u32 base=blockIdx.x*group_size, lane=threadIdx.x, core_pos=base+lane;
    const u32 pos=t.start-t.origin+core_pos;
    if (base>=t.core) return;
    if (t.core-base>=cooperative_span) {
        const u32 first=pos-lane, d=pred(p,first);
        if (d && d==pred(p,first+group_size-1) && d<=first && d<max_dist) {
            bool same=true;
            if (lane<cooperative_span/4) {
                const u32 offset=first+lane*4;
                same=words[offset]==words[offset-d];
            }
            if (__syncthreads_and(same) && pred(p,pos)==d) {
                output[core_pos]=(d<<16)|match_limit;
                return;
            }
        }
    }
    if (core_pos<t.core) output[core_pos]=match_word<true>(words,p,t.len,max_dist,budget,pos);
}

struct Symbol {
    u32 code, extra, value;
};
__device__ __forceinline__ Symbol length_symbol(u32 len) {
    if (len<=10) return {254+len,0,0};
    if (len==258) return {285,0,0};
    const u32 x=len-3, extra=29-__clz(x), part=(x>>extra)&3;
    return {261+4*extra+part,extra,x-((4+part)<<extra)};
}
__device__ __forceinline__ Symbol distance_symbol(u32 distance) {
    const u32 x=distance-1;
    if (x<4) return {x,0,0};
    const u32 extra=30-__clz(x), part=(x>>extra)&1;
    return {2+2*extra+part,extra,x-((2+part)<<extra)};
}
__device__ __forceinline__ u32 reverse_bits(u32 code,u32 len) {
    return len ? __brev(code)>>(32-len) : 0;
}
constexpr u32 ll_count=286, dd_count=30, cl_count=19;
struct Huffman {
    u16 codes[ll_count];
    unsigned char lengths[ll_count];
    u32 symbols;
};
__device__ __forceinline__ bool less_node(u32 a,u32 b,const u32* weights) {
    return weights[a]<weights[b] || (weights[a]==weights[b] && a<b);
}
__device__ void push_heap(u16* heap,u32& n,u16 node,const u32* weights) {
    u32 i=n++;
    while(i) {
        const u32 parent=(i-1)/2;
        if(!less_node(node,heap[parent],weights))break;
        heap[i]=heap[parent];i=parent;
    }
    heap[i]=node;
}
__device__ u16 pop_heap(u16* heap,u32& n,const u32* weights) {
    const u16 result=heap[0], last=heap[--n];
    u32 i=0;
    while(2*i+1<n) {
        u32 child=2*i+1;
        if(child+1<n && less_node(heap[child+1],heap[child],weights))++child;
        if(!less_node(heap[child],last,weights))break;
        heap[i]=heap[child];i=child;
    }
    if(n)heap[i]=last;
    return result;
}
// Exact canonical length-limited Huffman construction. In the uncommon
// overflow case, redistribute clipped lengths until Kraft equality holds,
// then assign longest lengths to the least frequent leaves.
__device__ void build_huffman(const u32* freq,u32 n,u32 max_bits,Huffman& out) {
    u32 weights[2*ll_count];
    u16 heap[ll_count], parent[2*ll_count], order[ll_count];
    u32 used=0, leaves=0;
    for(u32 i=0;i<n;++i) {
        out.lengths[i]=0;out.codes[i]=0;
        weights[i]=freq[i];
        if(freq[i]) {push_heap(heap,used,static_cast<u16>(i),weights);++leaves;}
    }
    for(u32 i=0;leaves<2 && i<n;++i) {
        if(!weights[i]){weights[i]=1;push_heap(heap,used,static_cast<u16>(i),weights);++leaves;}
    }
    u32 next=n, ordered=0;
    while(used>1) {
        const u16 a=pop_heap(heap,used,weights),b=pop_heap(heap,used,weights);
        if(a<n)order[ordered++]=a;
        if(b<n)order[ordered++]=b;
        weights[next]=weights[a]+weights[b];
        parent[a]=parent[b]=static_cast<u16>(next);
        push_heap(heap,used,static_cast<u16>(next++),weights);
    }
    const u32 root=heap[0];
    parent[root]=static_cast<u16>(root);
    u32 counts[16]={};
    u32 slots=0;
    for(u32 j=0;j<ordered;++j) {
        const u32 leaf=order[j];
        u32 len=0;
        for(u32 node=leaf;node!=root;node=parent[node])++len;
        len=min(len,max_bits);
        out.lengths[leaf]=static_cast<unsigned char>(len);
        ++counts[len];slots+=1u<<(max_bits-len);
    }
    const u32 target=1u<<max_bits;
    if(slots>target) {
        while(slots>target) {
            u32 bits=max_bits-1;
            while(!counts[bits])--bits;
            --counts[bits];counts[bits+1]+=2;--counts[max_bits];--slots;
        }
        u32 i=0;
        for(u32 bits=max_bits;bits;--bits)
            for(u32 j=0;j<counts[bits];++j) out.lengths[order[i++]]=static_cast<unsigned char>(bits);
    }
    u32 next_code[16]={},code=0;
    for(u32 bits=1;bits<=max_bits;++bits){code=(code+counts[bits-1])<<1;next_code[bits]=code;}
    out.symbols=0;
    for(u32 i=0;i<n;++i) {
        const u32 len=out.lengths[i];
        if(len){out.codes[i]=static_cast<u16>(reverse_bits(next_code[len]++,len));out.symbols=i+1;}
    }
}
__device__ __forceinline__ u32 fixed_length(u32 sym) {
    return sym<144 ? 8 : sym<256 ? 9 : sym<280 ? 7 : 8;
}
__device__ __forceinline__ u32 rle_extra(u32 sym) {
    return sym==16 ? 2 : sym==17 ? 3 : sym==18 ? 7 : 0;
}
__device__ __constant__ unsigned char cl_order[19]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
__device__ void add_rle(u16* runs,u32& count,u32* hist,u32 sym,u32 value=0) {
    runs[count++]=static_cast<u16>(sym|(value<<5));++hist[sym];
}
__device__ u32 header_rle(const Huffman& ll,const Huffman& dd,u32 nl,u32 nd,u16* runs,u32* hist) {
    unsigned char lengths[ll_count+dd_count];
    for(u32 i=0;i<nl;++i)lengths[i]=ll.lengths[i];
    for(u32 i=0;i<nd;++i)lengths[nl+i]=dd.lengths[i];
    const u32 n=nl+nd;
    u32 count=0;
    for(u32 i=0;i<n;) {
        const u32 value=lengths[i];
        u32 repeat=1;
        while(i+repeat<n && lengths[i+repeat]==value)++repeat;
        i+=repeat;
        if(!value) {
            while(repeat>=11){const u32 k=min(repeat,138u);add_rle(runs,count,hist,18,k-11);repeat-=k;}
            if(repeat>=3){const u32 k=min(repeat,10u);add_rle(runs,count,hist,17,k-3);repeat-=k;}
        } else {
            add_rle(runs,count,hist,value);--repeat;
            while(repeat>=3){const u32 k=min(repeat,6u);add_rle(runs,count,hist,16,k-3);repeat-=k;}
        }
        while(repeat){add_rle(runs,count,hist,value);--repeat;}
    }
    return count;
}
__device__ __forceinline__ u32 usable_match(u32 word) {
    return word&65535;
}
// Per-tile Adler components for the core only, with initial A=B=0.
// Use 64-bit weighted sums so all supported block sizes are exact.
extern "C" __global__ void resident_adler(const u32* input,u32* checksums,
    u32 size,u32 history,u32 tile_bytes,u32 max_dist) {
    const u32 tile=blockIdx.x;
    const Tile t=tile_at(tile,size,history,tile_bytes,max_dist);
    u64 a=0,b=0;
    for(u32 i=threadIdx.x;i<t.core;i+=blockDim.x) {
        const u32 value=input[t.start+i]&255;
        a+=value;b+=static_cast<u64>(t.core-i)*value;
    }
    for(u32 shift=16;shift;shift>>=1){a+=__shfl_down_sync(0xffffffffu,a,shift);b+=__shfl_down_sync(0xffffffffu,b,shift);}
    __shared__ u64 sa[32],sb[32];
    if(threadIdx.x%32==0){sa[threadIdx.x/32]=a;sb[threadIdx.x/32]=b;}
    __syncthreads();
    if(threadIdx.x==0) {
        a=b=0;
        for(u32 i=0;i<blockDim.x/32;++i){a+=sa[i];b+=sb[i];}
        checksums[2*tile]=static_cast<u32>(a%adler_mod);
        checksums[2*tile+1]=static_cast<u32>(b%adler_mod);
    }
}
// Reflected CRC-32 polynomial operations, for gzip framing. Contiguous
// thread chunks combine in a parallel reduction; host combines only windows.
constexpr u32 crc_poly=0xedb88320u;
__device__ u32 crc_multiply(u32 a,u32 b) {
    u32 result=0;
    for(u32 mask=0x80000000u;mask;mask>>=1) {
        if(a&mask)result^=b;
        if(!(a&(mask-1)))break;
        b=(b>>1)^((0u-(b&1))&crc_poly);
    }
    return result;
}
__device__ __constant__ u32 crc_shift_powers[32]={
    0x00800000u,0x00008000u,0xedb88320u,0xb1e6b092u,0xa06a2517u,0xed627daeu,0x88d14467u,0xd7bbfe6au,0xec447f11u,0x8e7ea170u,0x6427800eu,0x4d47bae0u,0x09fe548fu,0x83852d0fu,0x30362f1au,0x7b5a9cc3u,0x31fec169u,0x9fec022au,0x6c8dedc4u,0x15d6874du,0x5fde7a4eu,0xbad90e37u,0x2e4e5eefu,0x4eaba214u,0xa8a472c0u,0x429a969eu,0x148d302au,0xc40ba6d0u,0xc4e22c3cu,0x40000000u,0x20000000u,0x08000000u};
__device__ u32 crc_combine(u32 first,u32 second,u32 len) {
    u32 op=0x80000000u;
    for(u32 i=0;len;++i,len>>=1)if(len&1)op=crc_multiply(crc_shift_powers[i],op);
    return crc_multiply(op,first)^second;
}
__device__ __forceinline__ u32 crc_update(u32 crc,u32 byte) {
    crc^=byte;
    #pragma unroll
    for(u32 i=0;i<8;++i)crc=(crc>>1)^((0u-(crc&1))&crc_poly);
    return crc;
}
extern "C" __global__ void resident_crc(const u32* input,u32* checksums,
    u32 size,u32 history,u32 tile_bytes,u32 max_dist) {
    const u32 tile=blockIdx.x;
    const Tile t=tile_at(tile,size,history,tile_bytes,max_dist);
    const u32 chunk=(t.core+blockDim.x-1)/blockDim.x;
    const u32 start=min(t.core,threadIdx.x*chunk),end=min(t.core,start+chunk);
    u32 crc=0xffffffffu;
    for(u32 i=start;i<end;++i)crc=crc_update(crc,input[t.start+i]&255);
    __shared__ u32 crcs[128],lengths[128];
    crcs[threadIdx.x]=~crc;lengths[threadIdx.x]=end-start;
    __syncthreads();
    for(u32 stride=1;stride<blockDim.x;stride*=2) {
        if(threadIdx.x%(2*stride)==0) {
            const u32 j=threadIdx.x+stride;
            crcs[threadIdx.x]=crc_combine(crcs[threadIdx.x],crcs[j],lengths[j]);
            lengths[threadIdx.x]+=lengths[j];
        }
        __syncthreads();
    }
    if(!threadIdx.x){checksums[2*tile]=crcs[0];checksums[2*tile+1]=0;}
}
