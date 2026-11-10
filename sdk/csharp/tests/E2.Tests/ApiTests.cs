using Xunit;

namespace E2.Tests;

[Collection("model")]
public sealed class ApiTests
{
    private readonly ModelFixture _fixture;

    public ApiTests(ModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void LoadsTextTower()
    {
        if (_fixture.Model is not { } model) return;

        Assert.True((model.Modalities & E2Model.Modality.Text) != 0);
        Assert.NotEmpty(model.ModelInfo);
        Assert.Equal(E2Device.Cpu, model.Device);
        Assert.True(model.Threads >= 1);
    }

    [Fact]
    public void ThreadsRoundTrip()
    {
        if (_fixture.Model is not { } model) return;

        int before = model.Threads;
        try
        {
            model.Threads = 2;
            Assert.Equal(2, model.Threads);
        }
        finally
        {
            model.Threads = before;
        }
    }

    [Fact]
    public void LastMillisecondsIsReported()
    {
        if (_fixture.Model is not { } model) return;

        model.Embed("hello world");
        Assert.True(model.LastMilliseconds > 0);
    }

    [Fact]
    public void DisposeIsIdempotentAndRejectsUse()
    {
        if (!File.Exists(Repo.ModelPath)) return;

        var model = E2Model.Load(Repo.ModelPath, Repo.TokenizerPath);
        model.Dispose();
        model.Dispose();

        Assert.Throws<ObjectDisposedException>(() => { _ = model.Embed("hello"); });
        Assert.Throws<ObjectDisposedException>(() => { _ = model.ModelInfo; });
        Assert.Throws<ObjectDisposedException>(() => { _ = model.Modalities; });
        Assert.Throws<ObjectDisposedException>(() => { _ = model.Embed(new E2Input { Text = "hello" }); });
    }

    [Fact]
    public async Task CancelledTokenThrows()
    {
        if (_fixture.Model is not { } model) return;

        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => model.EmbedAsync("hello world", ct: cts.Token));
    }

    [Fact]
    public async Task PlaceholderCountMustMatch()
    {
        if (_fixture.Model is not { } model) return;

        await Assert.ThrowsAsync<ArgumentException>(
            () => model.EmbedAsync(new E2Input { Text = "a <|image|> b" }));
        await Assert.ThrowsAsync<ArgumentException>(
            () => model.EmbedAsync(new E2Input
            {
                Text = "no placeholders",
                Audio = [new float[16]],
            }));
    }

    [Fact]
    public async Task ConcurrentCallsAgreeWithSequential()
    {
        if (_fixture.Model is not { } model) return;

        var expected = await model.EmbedAsync("hello world");
        var tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(() => model.Embed("hello world")));
        foreach (var actual in await Task.WhenAll(tasks))
            Assert.True(Vec.Cosine(expected, actual) >= 0.999999);
    }

    [Fact]
    public async Task ConcurrentFailuresKeepTheirOwnErrorText()
    {
        if (_fixture.Model is not { } model) return;

        /* e2_error() is thread-local: a failure on one worker must not be
         * reported as another worker's message. Each task uses its own path. */
        var failures = Enumerable.Range(0, 8).Select(i => Task.Run(async () =>
        {
            var path = $"/nonexistent/worker-{i}.png";
            var e = await Assert.ThrowsAsync<E2Exception>(() => model.EmbedVideoAsync([path]));
            Assert.Contains($"worker-{i}.png", e.Message);
        }));
        await Task.WhenAll(failures);
    }

    [Fact]
    public void MissingModelFileThrows()
    {
        Assert.Throws<E2Exception>(() => E2Model.Load("/nonexistent/e2-model.bin"));
    }

    [Fact]
    public void DefaultTokenizerPathIsTheSiblingFile()
    {
        if (!File.Exists(Repo.ModelPath)) return;

        using var model = E2Model.Load(Repo.ModelPath);
        Assert.True((model.Modalities & E2Model.Modality.Text) != 0);
    }
}
