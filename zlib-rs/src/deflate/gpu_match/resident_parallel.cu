// Parallel entropy encoding for GPU-resident DEFLATE. Independent parse
// segments are grouped into entropy blocks by actual token count rather than
// decoded length; match indexing has its own independent tile geometry.
constexpr u32 hist_count=ll_count+dd_count+1;
struct EntropyBlock {
    u32 first,last,mode,header_bits,bytes;
    u32 ll[ll_count],dd[dd_count];
};
__device__ __forceinline__ u32 pack_code(const Huffman& h,u32 sym) {
    return h.codes[sym]|(static_cast<u32>(h.lengths[sym])<<16);
}
struct WordWriter {
    u32* output;
    u32 word,count;
    u64 pending=0;
    bool shared;
    __device__ WordWriter(u32* output,u32 bit):output(output),word(bit/32),count(bit%32),shared(count!=0) {}
    __device__ void put(u32 value,u32 bits) {
        pending|=static_cast<u64>(value)<<count;
        count+=bits;
        if(count>=32) {
            if(shared)atomicOr(output+word,static_cast<u32>(pending));
            else output[word]=static_cast<u32>(pending);
            ++word;pending>>=32;count-=32;shared=false;
        }
    }
    __device__ void code(u32 code) {put(code&65535,code>>16);}
    __device__ void finish() {
        if(count && pending)atomicOr(output+word,static_cast<u32>(pending));
    }
    __device__ u32 position() const {return word*32+count;}
};
constexpr u32 path_mark=0x80000000u;
constexpr u32 path_token_mask=~path_mark;
constexpr u32 path_states=match_limit;
__device__ __forceinline__ u32 advance_token(u32 token) {
    return token&0x7fff0000u ? token&65535u : 1u;
}
// Compute the lazy choice at every possible start before traversing its
// functional graph. Match distances use at most 15 bits in DEFLATE.
extern "C" __global__ void resident_choices(const u32* input,const u32* matches,u32* choices,
    u32 core,u32 history,u32 strategy,u32 offset) {
    const u32 pos=blockIdx.x*blockDim.x+threadIdx.x;
    if(pos>=core)return;
    const u32 match=matches[pos],raw_len=usable_match(match);
    const u32 raw_next=pos+1<core ? usable_match(matches[pos+1]) : 0;
    const u32 len=strategy==1 && raw_len<=5 ? 0 : raw_len;
    const u32 next=strategy==1 && raw_next<=5 ? 0 : raw_next;
    choices[pos]=len>=3 && (len==match_limit || next<=len)
        ? match : byte_at<false>(input,offset+history+pos);
}
// Mark a canonical path starting at each segment boundary. Each path
// exclusively modifies its own segment and may end up to 257 bytes beyond it.
extern "C" __global__ void resident_mark_paths(u32* choices,u32* exits,
    u32 core,u32 segment_bytes,u32 segments) {
    const u32 segment=blockIdx.x*blockDim.x+threadIdx.x;
    if(segment>=segments)return;
    const u32 start=segment*segment_bytes,end=min(core,start+segment_bytes);
    u32 pos=start;
    while(pos<end) {
        const u32 token=choices[pos];
        choices[pos]=token|path_mark;
        pos+=advance_token(token);
    }
    exits[segment]=pos-end;
}
// Compute all possible incoming-offset transitions in parallel. On
// coalescence with the canonical path, its already-known exit is exact.
// Literal runs merge immediately; long-match runs have few transitions.
extern "C" __global__ void resident_transitions(const u32* choices,const u32* exits,
    u16* transitions,u32 core,u32 segment_bytes) {
    const u32 segment=blockIdx.x,start=segment*segment_bytes,end=min(core,start+segment_bytes);
    for(u32 incoming=threadIdx.x;incoming<path_states;incoming+=blockDim.x) {
        u32 pos=min(end,start+incoming);
        while(pos<end) {
            const u32 token=choices[pos];
            if(token&path_mark){pos=end+exits[segment];break;}
            pos+=advance_token(token);
        }
        transitions[segment*path_states+incoming]=static_cast<u16>(pos-end);
    }
}
extern "C" __global__ void resident_scan_paths(const u16* transitions,u32* entries,u32 segments) {
    if(threadIdx.x||blockIdx.x)return;
    u32 offset=0;
    for(u32 segment=0;segment<segments;++segment) {
        entries[segment]=offset;
        offset=transitions[segment*path_states+offset];
    }
    entries[segments]=0;
}
extern "C" __global__ void resident_parse_paths(const u32* choices,u32* tokens_all,
    const u32* entries,u32* counts,u32 core,u32 segment_bytes,u32 segments) {
    const u32 segment=blockIdx.x*blockDim.x+threadIdx.x;
    if(segment>=segments)return;
    const u32 start=segment*segment_bytes,end=min(core,start+segment_bytes);
    u32* tokens=tokens_all+start;
    u32 count=0;
    for(u32 pos=start+entries[segment];pos<end;) {
        const u32 token=choices[pos]&path_token_mask;
        tokens[count++]=token;
        pos+=advance_token(token);
    }
    counts[segment]=count;
}
// Contiguous segments, bounded by both target token count and DEFLATE's
// useful maximum block span. Empty input is handled at the host boundary.
extern "C" __global__ void resident_partition(const u32* counts,u32* segment_blocks,
    EntropyBlock* blocks,u32* block_count,u32 segments,u32 target_tokens,u32 max_segments) {
    if(threadIdx.x || blockIdx.x)return;
    u32 block=0,first=0,tokens=0;
    for(u32 i=0;i<segments;++i) {
        if(i>first && counts[i] && (tokens+counts[i]>target_tokens || i-first>=max_segments)) {
            blocks[block].first=first;blocks[block].last=i;++block;first=i;tokens=0;
        }
        segment_blocks[i]=block;tokens+=counts[i];
    }
    blocks[block].first=first;blocks[block].last=segments;*block_count=block+1;
}
extern "C" __global__ void resident_clear(u32* words,u32 n) {
    for(u32 i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)words[i]=0;
}
extern "C" __global__ void resident_histogram(const u32* matches,const u32* counts,const u32* segment_blocks,
    u32* histograms,u32 segment_bytes) {
    const u32 segment=blockIdx.x;
    __shared__ u32 histogram[hist_count];
    for(u32 i=threadIdx.x;i<hist_count;i+=blockDim.x)histogram[i]=0;
    __syncthreads();
    const u32* tokens=matches+segment*segment_bytes;
    u32 extra=0;
    for(u32 i=threadIdx.x;i<counts[segment];i+=blockDim.x) {
        const u32 token=tokens[i];
        if(token>>16) {
            const Symbol l=length_symbol(token&65535),d=distance_symbol(token>>16);
            atomicAdd(histogram+l.code,1);atomicAdd(histogram+ll_count+d.code,1);
            extra+=l.extra+d.extra;
        } else atomicAdd(histogram+token,1);
    }
    if(extra)atomicAdd(histogram+hist_count-1,extra);
    __syncthreads();
    u32* dest=histograms+segment_blocks[segment]*hist_count;
    for(u32 i=threadIdx.x;i<hist_count;i+=blockDim.x)if(histogram[i])atomicAdd(dest+i,histogram[i]);
}
__device__ __forceinline__ u32 actual_start(u32 segment,const u32* entries,u32 core,u32 segment_bytes) {
    return min(core,segment*segment_bytes+entries[segment]);
}
__device__ __forceinline__ u32 block_core(const EntropyBlock& block,const u32* entries,u32 core,u32 segment_bytes) {
    return actual_start(block.last,entries,core,segment_bytes)-actual_start(block.first,entries,core,segment_bytes);
}
__device__ __forceinline__ u32 fixed_code(u32 sym) {
    const u32 code=sym<144 ? 48+sym : sym<256 ? 400+sym-144 : sym<280 ? sym-256 : 192+sym-280;
    const u32 bits=fixed_length(sym);
    return reverse_bits(code,bits)|(bits<<16);
}
__device__ void stored_header(unsigned char* output,u32 pos,u32 len,bool final) {
    output[pos]=static_cast<unsigned char>(final);
    output[pos+1]=static_cast<unsigned char>(len);
    output[pos+2]=static_cast<unsigned char>(len>>8);
    output[pos+3]=static_cast<unsigned char>(~len);
    output[pos+4]=static_cast<unsigned char>((~len)>>8);
}
// One thread per entropy block builds its trees and exact encoded size.
// The header and trailer are written first, then independent segments fill
// the non-overlapping bit ranges between them.
extern "C" __global__ void resident_tree(EntropyBlock* blocks,const u32* block_count,u32* histograms,
    u32* output,const u32* entries,u32 core,u32 segment_bytes,u32 arena_stride,u32 final_window,u32 strategy) {
    const u32 index=blockIdx.x*blockDim.x+threadIdx.x,n=*block_count;
    if(index>=n)return;
    EntropyBlock& block=blocks[index];
    u32* ll_freq=histograms+index*hist_count;
    u32* dd_freq=ll_freq+ll_count;
    ++ll_freq[256];
    const u32 extra=dd_freq[dd_count];
    Huffman ll,dd,cl;
    build_huffman(ll_freq,ll_count,15,ll);build_huffman(dd_freq,dd_count,15,dd);
    const u32 nl=max(ll.symbols,257u),nd=dd.symbols;
    u16 runs[ll_count+dd_count];
    u32 cl_freq[cl_count]={};
    const u32 nr=header_rle(ll,dd,nl,nd,runs,cl_freq);
    build_huffman(cl_freq,cl_count,7,cl);
    u32 nc=cl_count;
    while(nc>4 && !cl.lengths[cl_order[nc-1]])--nc;
    u32 dynamic_bits=3+5+5+4+3*nc+extra,fixed_bits=3+extra;
    for(u32 i=0;i<cl_count;++i)dynamic_bits+=cl_freq[i]*(cl.lengths[i]+rle_extra(i));
    for(u32 i=0;i<ll_count;++i){dynamic_bits+=ll_freq[i]*ll.lengths[i];fixed_bits+=ll_freq[i]*fixed_length(i);}
    for(u32 i=0;i<dd_count;++i){dynamic_bits+=dd_freq[i]*dd.lengths[i];fixed_bits+=dd_freq[i]*5;}
    const bool final=final_window && index+1==n;
    const u32 align_bits=final ? 0 : 3,align_bytes=final ? 0 : 4;
    const u32 dynamic_bytes=strategy==4 ? 0xffffffffu : (dynamic_bits+align_bits+7)/8+align_bytes;
    const u32 fixed_bytes=(fixed_bits+align_bits+7)/8+align_bytes;
    const u32 len=block_core(block,entries,core,segment_bytes);
    const u32 stored_bytes=len+5*((len+65534)/65535);
    u32* dest=output+static_cast<u64>(block.first)*(arena_stride/4);
    if(stored_bytes<=min(dynamic_bytes,fixed_bytes)) {
        block.mode=0;block.bytes=stored_bytes;block.header_bits=0;
        for(u32 pos=0;pos<len;) {
            const u32 chunk=min(65535u,len-pos);
            stored_header(reinterpret_cast<unsigned char*>(dest),pos+5*(pos/65535),chunk,final&&pos+chunk==len);
            pos+=chunk;
        }
        return;
    }
    const bool dynamic=dynamic_bytes<=fixed_bytes;
    block.mode=dynamic ? 2 : 1;block.bytes=dynamic ? dynamic_bytes : fixed_bytes;
    for(u32 i=0;i<ll_count;++i)block.ll[i]=dynamic ? pack_code(ll,i) : fixed_code(i);
    for(u32 i=0;i<dd_count;++i)block.dd[i]=dynamic ? pack_code(dd,i) : reverse_bits(i,5)|(5u<<16);
    WordWriter header(dest,0);
    header.put((dynamic ? 4u : 2u)|static_cast<u32>(final),3);
    if(dynamic) {
        header.put(nl-257,5);header.put(nd-1,5);header.put(nc-4,4);
        for(u32 i=0;i<nc;++i)header.put(cl.lengths[cl_order[i]],3);
        for(u32 i=0;i<nr;++i) {
            const u32 sym=runs[i]&31;
            header.code(pack_code(cl,sym));header.put(runs[i]>>5,rle_extra(sym));
        }
    }
    block.header_bits=header.position();
    header.finish();
    // Start exactly at the last token's end, then add EOB and alignment.
    const u32 end=(dynamic ? dynamic_bits : fixed_bits)-(block.ll[256]>>16);
    WordWriter tail(dest,end);
    tail.code(block.ll[256]);
    if(!final) {
        tail.put(0,3);
        const u32 pad=(8-tail.position()%8)%8;
        tail.put(0,pad);tail.put(0,16);tail.put(65535,16);
    }
    tail.finish();
}
__device__ __forceinline__ u32 token_bits(const EntropyBlock& block,u32 token) {
    if(!(token>>16))return block.ll[token]>>16;
    const Symbol l=length_symbol(token&65535),d=distance_symbol(token>>16);
    return (block.ll[l.code]>>16)+(block.dd[d.code]>>16)+l.extra+d.extra;
}
extern "C" __global__ void resident_segment_bits(const u32* matches,const u32* counts,
    const u32* segment_blocks,const EntropyBlock* blocks,u32* bits,u32 segment_bytes) {
    const u32 segment=blockIdx.x;
    const EntropyBlock& block=blocks[segment_blocks[segment]];
    if(!block.mode){if(!threadIdx.x)bits[segment]=0;return;}
    const u32* tokens=matches+segment*segment_bytes;
    u32 sum=0;
    for(u32 i=threadIdx.x;i<counts[segment];i+=blockDim.x)sum+=token_bits(block,tokens[i]);
    for(u32 shift=16;shift;shift>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,shift);
    __shared__ u32 warp_sums[32];
    if(threadIdx.x%32==0)warp_sums[threadIdx.x/32]=sum;
    __syncthreads();
    if(!threadIdx.x){sum=0;for(u32 i=0;i<blockDim.x/32;++i)sum+=warp_sums[i];bits[segment]=sum;}
}
extern "C" __global__ void resident_bit_offsets(const EntropyBlock* blocks,const u32* block_count,
    const u32* bits,u32* offsets) {
    const u32 index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=*block_count)return;
    const EntropyBlock& block=blocks[index];
    u32 pos=block.header_bits;
    for(u32 i=block.first;i<block.last;++i){offsets[i]=pos;pos+=bits[i];}
}
extern "C" __global__ void resident_write_segments(const u32* matches,const u32* counts,
    const u32* segment_blocks,const EntropyBlock* blocks,const u32* offsets,u32* output,
    u32 segment_bytes,u32 arena_stride,u32 segments) {
    const u32 segment=blockIdx.x*blockDim.x+threadIdx.x;
    if(segment>=segments)return;
    const EntropyBlock& block=blocks[segment_blocks[segment]];
    u32* dest=output+static_cast<u64>(block.first)*(arena_stride/4);
    if(!block.mode)return;
    const u32* tokens=matches+segment*segment_bytes;
    WordWriter w(dest,offsets[segment]);
    for(u32 i=0;i<counts[segment];++i) {
        const u32 token=tokens[i];
        if(token>>16) {
            const Symbol l=length_symbol(token&65535),d=distance_symbol(token>>16);
            w.code(block.ll[l.code]);w.put(l.value,l.extra);
            w.code(block.dd[d.code]);w.put(d.value,d.extra);
        } else w.code(block.ll[token]);
    }
    w.finish();
}
// Stored payload is bulk-copied in parallel, excluding its five-byte headers.
extern "C" __global__ void resident_write_stored(const u32* input,const u32* segment_blocks,
    const EntropyBlock* blocks,u32* output,const u32* entries,u32 core,u32 history,u32 segment_bytes,u32 arena_stride,u32 offset) {
    const u32 segment=blockIdx.x;
    const EntropyBlock& block=blocks[segment_blocks[segment]];
    if(block.mode)return;
    unsigned char* dest=reinterpret_cast<unsigned char*>(output)+static_cast<u64>(block.first)*arena_stride;
    const u32 start=actual_start(segment,entries,core,segment_bytes);
    const u32 len=actual_start(segment+1,entries,core,segment_bytes)-start;
    const u32 block_start=actual_start(block.first,entries,core,segment_bytes);
    for(u32 i=threadIdx.x;i<len;i+=blockDim.x) {
        const u32 pos=start-block_start+i;
        dest[pos+5*(pos/65535+1)]=static_cast<unsigned char>(byte_at<false>(input,offset+history+start+i));
    }
}
extern "C" __global__ void resident_parallel_offsets(const EntropyBlock* blocks,const u32* block_count,
    const u32* checksums,u32* offsets,u32* meta,u32 core,u32 checksum_tile,u32 checksum_tiles,u32 checksum_kind) {
    if(threadIdx.x || blockIdx.x)return;
    u32 total=0,a=0,b=0;
    for(u32 i=0;i<*block_count;++i){offsets[i]=total;total+=blocks[i].bytes;}
    for(u32 i=0;i<checksum_tiles;++i) {
        const u32 n=min(checksum_tile,core-i*checksum_tile);
        if(checksum_kind==2)a=crc_combine(a,checksums[2*i],n);
        else if(checksum_kind==1) {
            b=static_cast<u32>((b+checksums[2*i+1]+static_cast<u64>(n)*a)%adler_mod);
            a=(a+checksums[2*i])%adler_mod;
        }
    }
    meta[0]=total;meta[1]=a;meta[2]=b;meta[3]=*block_count;
}
extern "C" __global__ void resident_parallel_compact(const EntropyBlock* blocks,const u32* block_count,
    const u32* offsets,const unsigned char* arena,unsigned char* compact,u32 arena_stride) {
    const u32 index=blockIdx.x;
    if(index>=*block_count)return;
    const EntropyBlock& block=blocks[index];
    const unsigned char* src=arena+static_cast<u64>(block.first)*arena_stride;
    unsigned char* dest=compact+offsets[index];
    for(u32 i=threadIdx.x;i<block.bytes;i+=blockDim.x)dest[i]=src[i];
}
