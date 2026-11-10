using Xunit;

namespace E2.Tests;

[Collection("model")]
public sealed class TextTests
{
    private readonly ModelFixture _fixture;

    public TextTests(ModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void TextMatchesGolden()
    {
        if (_fixture.Model is not { } model) return;

        int seen = 0;
        foreach (var (name, expected) in Golden.Entries)
        {
            if (!name.StartsWith("text_default:", StringComparison.Ordinal)) continue;
            var text = name["text_default:".Length..];
            var actual = model.Embed(text);
            double cos = Vec.Cosine(expected, actual);
            Assert.True(cos >= 0.99999, $"'{text}': cosine {cos:F8}");
            seen++;
        }
        Assert.True(seen > 0, "no text entries in the golden file");
    }

    [Fact]
    public void TaskPrefixMatchesGolden()
    {
        if (_fixture.Model is not { } model) return;

        int seen = 0;
        foreach (var (name, expected) in Golden.Entries)
        {
            if (!name.StartsWith("task:", StringComparison.Ordinal)) continue;
            var rest = name["task:".Length..];
            int split = rest.IndexOf(':');
            var task = rest[..split];
            var text = rest[(split + 1)..];
            var actual = model.Embed(task, text);
            double cos = Vec.Cosine(expected, actual);
            Assert.True(cos >= 0.99999, $"{task}/{text}: cosine {cos:F8}");
            seen++;
        }
        Assert.True(seen > 0, "no task entries in the golden file");
    }

    [Theory]
    [InlineData(EmbeddingDimension.D128)]
    [InlineData(EmbeddingDimension.D256)]
    [InlineData(EmbeddingDimension.D512)]
    [InlineData(EmbeddingDimension.D768)]
    public async Task DimensionsAreNormalizedPrefixes(EmbeddingDimension dim)
    {
        if (_fixture.Model is not { } model) return;

        var full = await model.EmbedAsync("hello world");
        var small = await model.EmbedAsync("hello world", dim);

        Assert.Equal((int)dim, small.Length);
        Assert.Equal(1.0, Vec.Norm(small), 4);
        Assert.True(Vec.Cosine(Vec.Normalized(full.AsSpan(0, (int)dim)), small) >= 0.999999);
    }

    [Fact]
    public async Task BatchMatchesSingle()
    {
        if (_fixture.Model is not { } model) return;

        string[] texts = ["hello world", "a brown fox jumps over the lazy dog", "one more"];
        var batch = await model.EmbedBatchAsync(texts);

        Assert.Equal(texts.Length, batch.Length);
        for (int i = 0; i < texts.Length; i++)
        {
            var single = await model.EmbedAsync(texts[i]);
            Assert.True(Vec.Cosine(single, batch[i]) >= 0.999999, $"batch[{i}] differs from single");
        }
    }

    [Fact]
    public async Task EmptyBatchIsEmpty()
    {
        if (_fixture.Model is not { } model) return;
        Assert.Empty(await model.EmbedBatchAsync([]));
    }

    [Fact]
    public async Task SyncMatchesAsync()
    {
        if (_fixture.Model is not { } model) return;

        var sync = model.Embed("hello world");
        var async = await model.EmbedAsync("hello world");
        Assert.Equal(sync, async);
    }

    [Fact]
    public void InvalidDimensionThrows()
    {
        if (_fixture.Model is not { } model) return;
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = model.Embed("hello", (EmbeddingDimension)100); });
    }
}
