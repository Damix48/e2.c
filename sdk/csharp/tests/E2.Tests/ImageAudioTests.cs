using Xunit;

namespace E2.Tests;

[Collection("model")]
public sealed class ImageAudioTests
{
    private readonly ModelFixture _fixture;

    public ImageAudioTests(ModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void ImageRgbMatchesGolden()
    {
        if (_fixture.Model is not { } model) return;
        if ((model.Modalities & E2Model.Modality.Image) == 0) return;
        if (!Golden.Entries.TryGetValue("image:rgb128x128", out var expected)) return;

        var actual = model.Embed(Synth.Rgb(), Synth.ImageWidth, Synth.ImageHeight);
        double cos = Vec.Cosine(expected, actual);
        Assert.True(cos >= 0.99999, $"image cosine {cos:F8}");
    }

    [Fact]
    public void AudioMatchesGolden()
    {
        if (_fixture.Model is not { } model) return;
        if ((model.Modalities & E2Model.Modality.Audio) == 0) return;
        if (!Golden.Entries.TryGetValue("audio:square4s", out var expected)) return;

        var actual = model.Embed(Synth.Square4s());
        double cos = Vec.Cosine(expected, actual);
        Assert.True(cos >= 0.99999, $"audio cosine {cos:F8}");
    }

    [Fact]
    public async Task AudioFileUsesTheSamePathAsSamples()
    {
        if (_fixture.Model is not { } model) return;

        var pcm = Synth.Square4s();
        var path = Wav.WritePcm16(Path.GetTempPath(), pcm);
        try
        {
            var fromSamples = await model.EmbedAsync(pcm);
            float[] fromFile;
            try
            {
                fromFile = await model.EmbedAsync(new FileInfo(path));
            }
            catch (E2Exception e) when (e.Message.Contains("libsndfile"))
            {
                return; /* vendored build: no audio file decoding */
            }
            Assert.True(Vec.Cosine(fromSamples, fromFile) >= 0.9999, "wav file and raw samples disagree");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void UnsupportedExtensionThrows()
    {
        if (_fixture.Model is not { } model) return;
        var path = Path.Combine(Path.GetTempPath(), "e2-not-a-media-file.txt");
        File.WriteAllText(path, "hello");
        try
        {
            Assert.Throws<NotSupportedException>(() => model.Embed(new FileInfo(path)));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ShortRgbThrows()
    {
        if (_fixture.Model is not { } model) return;
        Assert.Throws<ArgumentException>(() => model.Embed(new byte[10], 4, 4));
    }

    [Fact]
    public async Task MissingVideoFrameSurfacesAsE2Exception()
    {
        if (_fixture.Model is not { } model) return;

        var e = await Assert.ThrowsAsync<E2Exception>(() =>
            model.EmbedVideoAsync([Path.Combine(Path.GetTempPath(), "e2-no-such-frame.png")]));
        Assert.NotEmpty(e.Message);
    }
}
