namespace E2;

/// <summary>
/// Output size. The model is trained for Matryoshka truncation, so a smaller
/// value is a prefix of the 768-dimensional embedding, re-normalized.
/// </summary>
public enum EmbeddingDimension
{
    D128 = 128,
    D256 = 256,
    D512 = 512,
    D768 = 768,
}

/// <summary>Compute device. CPU is the default and the reference path.</summary>
public enum E2Device
{
    Cpu = 0,
    Gpu = 1,
}
