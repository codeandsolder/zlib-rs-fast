#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn is_enabled_pclmulqdq() -> bool {
    #[cfg(feature = "std")]
    {
        return std::is_x86_feature_detected!("pclmulqdq")
            && std::is_x86_feature_detected!("sse4.1");
    }

    #[cfg(not(feature = "std"))]
    {
        false
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn is_enabled_crc() -> bool {
    #[cfg(feature = "std")]
    {
        return std::arch::is_aarch64_feature_detected!("crc");
    }

    #[cfg(not(feature = "std"))]
    {
        false
    }
}
