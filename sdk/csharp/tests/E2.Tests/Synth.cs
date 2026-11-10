namespace E2.Tests;

/* Inputs that tools/e2_golden.c produces natively. Keep them integer-derived
 * so both sides build bit-identical buffers. */
internal static class Synth
{
    public const int ImageWidth = 128;
    public const int ImageHeight = 128;

    public static byte[] Rgb()
    {
        var rgb = new byte[ImageWidth * ImageHeight * 3];
        for (int y = 0; y < ImageHeight; y++)
            for (int x = 0; x < ImageWidth; x++)
            {
                int o = (y * ImageWidth + x) * 3;
                rgb[o + 0] = (byte)((x * 7 + y * 13) & 255);
                rgb[o + 1] = (byte)((x * 11 + y * 3) & 255);
                rgb[o + 2] = (byte)((x * 5 + y * 17) & 255);
            }
        return rgb;
    }

    public static float[] Square4s()
    {
        var pcm = new float[64000]; /* 4 s at 16 kHz */
        /* / 512: exact, so the buffer matches tools/e2_golden.c bit for bit. */
        for (int i = 0; i < pcm.Length; i++) pcm[i] = ((i % 200) - 100) / 512f;
        return pcm;
    }
}

internal static class Wav
{
    /* 16-bit PCM mono; enough for libsndfile and for the vendored reader. */
    public static string WritePcm16(string directory, ReadOnlySpan<float> mono, int sampleRate = 16000)
    {
        var path = Path.Combine(directory, $"e2-{Guid.NewGuid():N}.wav");
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write("RIFF"u8);
        writer.Write(36 + mono.Length * 2);
        writer.Write("WAVE"u8);
        writer.Write("fmt "u8);
        writer.Write(16);
        writer.Write((short)1);          /* PCM */
        writer.Write((short)1);          /* mono */
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);    /* byte rate */
        writer.Write((short)2);          /* block align */
        writer.Write((short)16);         /* bits */
        writer.Write("data"u8);
        writer.Write(mono.Length * 2);
        foreach (var s in mono)
            writer.Write((short)Math.Clamp((int)MathF.Round(s * 32767f), -32768, 32767));
        return path;
    }
}
