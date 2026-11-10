namespace E2;

/// <summary>
/// Interleaved multimodal input. <see cref="Text"/> may contain
/// <c>&lt;|image|&gt;</c>, <c>&lt;|audio|&gt;</c> and <c>&lt;|video|&gt;</c>
/// placeholders; each one consumes the next entry of the matching list.
/// </summary>
public sealed class E2Input
{
    public string Text { get; init; } = "";

    /// <summary>Image file paths, consumed left to right by the placeholders.</summary>
    public IReadOnlyList<string> Images { get; init; } = [];

    /// <summary>One entry per video placeholder; each entry is that video's frame paths.</summary>
    public IReadOnlyList<IReadOnlyList<string>> Videos { get; init; } = [];

    /// <summary>One entry per audio placeholder: mono float32 PCM at 16 kHz.</summary>
    public IReadOnlyList<ReadOnlyMemory<float>> Audio { get; init; } = [];
}
