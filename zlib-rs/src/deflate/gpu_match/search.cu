// CUDA specialization of the finite-budget matcher in gpu_match.rs.
// Distances are unsigned 16-bit predecessors; results use the same packed ABI.
// Every logical input position owns its result word.
using u32 = unsigned int;
constexpr u32 match_limit = MATCH_LIMIT;
constexpr u32 group_size = GROUP_SIZE;
constexpr u32 cooperative_span = (group_size + match_limit - 1 + 3) & ~3u;

template<bool aligned> __device__ __forceinline__
u32 word(const u32* input, u32 index) {
    if constexpr (aligned) return input[index];
    const u32 i = index >> 2, shift = (index & 3) * 8;
    if (!shift) return input[i];
    return __funnelshift_r(input[i], input[i+1], shift);
}
template<bool aligned> __device__ __forceinline__
u32 byte_at(const u32* input, u32 index) {
    if constexpr (aligned) return input[index] & 255;
    return (input[index >> 2] >> ((index & 3) * 8)) & 255;
}
__device__ __forceinline__ u32 pred(const u32* p, u32 index) {
    return (p[index >> 1] >> ((index & 1) * 16)) & 65535;
}
template<bool aligned> __device__ __forceinline__
u32 common_length(const u32* input, u32 pos, u32 candidate, u32 limit, u32 probe) {
    u32 len = 0;
    if (limit >= 4) {
        const u32 diff = probe ^ word<aligned>(input, candidate);
        if (diff) return (__ffs(diff) - 1) >> 3;
        len = 4;
    }
    while (limit - len >= 16) {
        #pragma unroll
        for (u32 offset = 0; offset < 16; offset += 4) {
            const u32 diff = word<aligned>(input,pos+len+offset) ^ word<aligned>(input,candidate+len+offset);
            if (diff) return len + offset + ((__ffs(diff)-1)>>3);
        }
        len += 16;
    }
    while (limit-len >= 4) {
        const u32 diff = word<aligned>(input,pos+len) ^ word<aligned>(input,candidate+len);
        if (diff) return len + ((__ffs(diff)-1)>>3);
        len += 4;
    }
    while (len < limit && byte_at<aligned>(input,pos+len) == byte_at<aligned>(input,candidate+len)) ++len;
    return len;
}
__device__ __forceinline__ u32 shifted_offset(const u32* p, u32 candidate, u32 len) {
    const u32 tail = len-3, middle = tail/2;
    u32 offset = 0, distance = pred(p,candidate);
    if (!distance) return 0;
    const u32 tail_distance = pred(p,candidate+tail);
    if (!tail_distance) return tail;
    if (tail_distance > distance) { offset=tail; distance=tail_distance; }
    const u32 middle_distance=pred(p,candidate+middle);
    return (!middle_distance || middle_distance>distance) ? middle : offset;
}
template<bool aligned> __device__ __forceinline__
void search(const u32* input, const u32* p, u32* output, u32 size, u32 max_dist, u32 cooperative, u32 budget) {
    const u32 base=blockIdx.x*group_size, lane=threadIdx.x;
    const u32 pos=base+lane;
    if (cooperative && size-base >= cooperative_span) {
        // All lanes compute the same two cheap loads; vote over the shared span.
        const u32 distance=pred(p,base);
        if (distance && distance==pred(p,base+group_size-1) && distance<=base && distance<max_dist) {
            bool same=true;
            if (lane < cooperative_span/4) {
                const u32 offset=base+lane*4;
                same=word<aligned>(input,offset)==word<aligned>(input,offset-distance);
            }
            if (__syncthreads_and(same) && pred(p,pos)==distance) {
                output[pos]=(distance<<16)|match_limit;
                return;
            }
        }
    }
    if (pos>=size) return;
    if (size-pos<3) {output[pos]=0; return;}
    const u32 minimum=pos>=max_dist ? pos-max_dist+1 : 0;
    const u32 limit=min(match_limit,size-pos);
    const u32 probe=limit>=4 ? word<aligned>(input,pos) : 0;
    u32 cursor=pos, chain_offset=0, distance=pred(p,pos);
    u32 best_len=0,best_distance=0,best_probe=probe;
    for (u32 visited=0; best_len<limit && distance && visited<budget; ++visited) {
        if (distance>cursor) break;
        const u32 candidate=cursor-distance;
        if (candidate<minimum) break;
        cursor=candidate;
        distance=pred(p,cursor+chain_offset);
        if (best_len>=3) {
            if constexpr (aligned) {
                if (best_probe!=word<aligned>(input,candidate+best_len-3)) continue;
            } else if (byte_at<aligned>(input,pos+best_len)!=byte_at<aligned>(input,candidate+best_len)) continue;
        }
        const u32 len=common_length<aligned>(input,pos,candidate,limit,probe);
        if (len>=3 && len>best_len) {
            best_len=len;
            best_distance=pos-candidate;
            if (best_len<limit) {
                if (best_len>3) {
                    chain_offset=shifted_offset(p,candidate,best_len);
                    distance=pred(p,candidate+chain_offset);
                }
                if (distance) {
                    const u32 extension_offset=best_len-2;
                    const u32 extension_distance=pred(p,pos+extension_offset);
                    if (!extension_distance) distance=0;
                    else if (extension_distance>best_distance+distance) {
                        chain_offset=extension_offset;
                        distance=extension_distance-best_distance;
                    }
                }
                if constexpr (aligned) {
                    if (best_len>3) best_probe=word<aligned>(input,pos+best_len-3);
                }
            }
        }
    }
    output[pos]=(best_distance<<16)|best_len;
}
extern "C" __global__ void search_packed(const u32* input,const u32* p,u32* out,u32 size,u32 distance,u32 cooperative,u32 budget) {
    search<false>(input,p,out,size,distance,cooperative,budget);
}
extern "C" __global__ void search_aligned(const u32* input,const u32* p,u32* out,u32 size,u32 distance,u32 cooperative,u32 budget) {
    search<true>(input,p,out,size,distance,cooperative,budget);
}
