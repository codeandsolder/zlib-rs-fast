#[inline(always)]
pub fn is_enabled_pclmulqdq() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::is_x86_feature_detected!("pclmulqdq") && std::is_x86_feature_detected!("sse4.1")
    }

    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    {
        false
    }
}

#[inline(always)]
pub fn is_enabled_crc() -> bool {
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    {
        std::arch::is_aarch64_feature_detected!("crc")
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "std")))]
    {
        false
    }
}
