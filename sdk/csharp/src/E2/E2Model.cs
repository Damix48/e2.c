using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace E2;

/// <summary>
/// One loaded EmbeddingGemma-2 model. Thread-safe, but calls are serialized:
/// the native engine has a scratch arena and its own thread pool, so it runs
/// one forward at a time. For concurrent requests, load one model per request
/// — the weights are read-only mmap, so extra contexts share the page cache.
/// </summary>
public sealed class E2Model : IDisposable
{
    private static readonly HashSet<string> ImageExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg", ".webp" };

    private static readonly HashSet<string> AudioExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".wav", ".flac", ".ogg", ".opus", ".mp3", ".aiff", ".aif" };

    private readonly ContextHandle _ctx;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private volatile bool _disposed;

    private E2Model(ContextHandle ctx) => _ctx = ctx;

    /// <summary>
    /// Loads a model file. <paramref name="tokenizerPath"/> defaults to
    /// <c>e2-tokenizer.bin</c> next to the model file.
    /// </summary>
    public static E2Model Load(string modelPath, string? tokenizerPath = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        tokenizerPath ??= Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath))!, "e2-tokenizer.bin");

        E2Library.EnsureAvailable();
        var handle = NativeMethods.e2_load(modelPath, tokenizerPath);
        if (handle == IntPtr.Zero)
            throw new E2Exception(
                $"e2_load({modelPath}): {Marshal.PtrToStringUTF8(NativeMethods.e2_error())}");
        return new E2Model(new ContextHandle(handle));
    }

    /// <summary>Towers present in the model file.</summary>
    [Flags]
    public enum Modality
    {
        None = 0,
        Text = NativeMethods.DevText,
        Image = NativeMethods.DevImage,
        Audio = NativeMethods.DevAudio,
        Video = NativeMethods.DevVideo,
    }

    public Modality Modalities => (Modality)NativeMethods.e2_modalities(Handle);

    /// <summary>Threads used by the engine. 0 = one per physical core.</summary>
    public int Threads
    {
        get => NativeMethods.e2_threads(Handle);
        set => Exclusive(h => NativeMethods.e2_set_threads(h, value));
    }

    /// <summary>Compute device. Setting <see cref="E2Device.Gpu"/> throws when the GPU is unavailable.</summary>
    public E2Device Device
    {
        get => (E2Device)NativeMethods.e2_device(Handle);
        set => Exclusive(h =>
        {
            if (NativeMethods.e2_set_device(h, (int)value) != 0)
                throw new E2Exception($"e2_set_device({value}): device unavailable, staying on the CPU");
        });
    }

    /// <summary>Model summary string from the native engine.</summary>
    public string ModelInfo => Marshal.PtrToStringUTF8(NativeMethods.e2_model_info(Handle)) ?? "";

    /// <summary>Wall time of the last embed call, in milliseconds.</summary>
    public double LastMilliseconds => NativeMethods.e2_last_ms(Handle);

    /* ------------------------------------------------------------------ */

    /// <summary>Embeds text with the model's default (no task prefix).</summary>
    public unsafe Task<float[]> EmbedAsync(string text, EmbeddingDimension dim = EmbeddingDimension.D768,
                                    CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        int d = Size(dim);
        return EmbedWith(d, (ctx, p) =>
            E2Library.Check(NativeMethods.e2_embed_text(ctx, text, p, d), "e2_embed_text"), ct);
    }

    /// <summary>Embeds text behind a task prefix, e.g. <c>search_query</c>, <c>search_document</c>.</summary>
    public unsafe Task<float[]> EmbedAsync(string task, string text,
                                    EmbeddingDimension dim = EmbeddingDimension.D768,
                                    CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(text);
        int d = Size(dim);
        return EmbedWith(d, (ctx, p) =>
            E2Library.Check(NativeMethods.e2_embed_text_task(ctx, task, text, p, d), "e2_embed_text_task"), ct);
    }

    /// <summary>Embeds a batch as one native call — faster than one call per text.</summary>
    public unsafe Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts,
                                          EmbeddingDimension dim = EmbeddingDimension.D768,
                                          CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        int d = Size(dim);
        if (texts.Count == 0) return Task.FromResult(Array.Empty<float[]>());

        return GatedAsync(ctx =>
        {
            using var arena = new UnmanagedArena();
            var ptrs = (IntPtr*)arena.PointerArray(texts.Count);
            for (int i = 0; i < texts.Count; i++) ptrs[i] = arena.Utf8(texts[i]);

            var flat = new float[texts.Count * d];
            fixed (float* p = flat)
                E2Library.Check(NativeMethods.e2_embed_text_batch(ctx, ptrs, texts.Count, p, d),
                                "e2_embed_text_batch");

            var result = new float[texts.Count][];
            for (int i = 0; i < texts.Count; i++) result[i] = flat.AsSpan(i * d, d).ToArray();
            return result;
        }, ct);
    }

    /// <summary>
    /// Embeds an image or audio file. The type is chosen from the extension:
    /// png/jpg/jpeg/webp as an image, wav/flac/ogg/opus/mp3/aiff as audio.
    /// </summary>
    public unsafe Task<float[]> EmbedAsync(FileInfo file, EmbeddingDimension dim = EmbeddingDimension.D768,
                                    CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(file);
        int d = Size(dim);
        var path = file.FullName;

        if (ImageExtensions.Contains(file.Extension))
            return EmbedWith(d, (ctx, p) =>
                E2Library.Check(NativeMethods.e2_embed_image(ctx, path, p, d), "e2_embed_image"), ct);

        if (AudioExtensions.Contains(file.Extension))
            return GatedAsync(ctx =>
            {
                E2Library.Check(NativeMethods.e2_audio_read(path, out var samples, out int n),
                                $"e2_audio_read({file.Name})");
                try
                {
                    var buf = new float[d];
                    fixed (float* p = buf)
                        E2Library.Check(NativeMethods.e2_embed_audio(ctx, (float*)samples, n, p, d),
                                        "e2_embed_audio");
                    return buf;
                }
                finally
                {
                    NativeMethods.e2_free_buffer(samples);
                }
            }, ct);

        throw new NotSupportedException(
            $"e2: unsupported file type '{file.Extension}'. Images: " +
            $"{string.Join(", ", ImageExtensions)}. Audio: {string.Join(", ", AudioExtensions)}.");
    }

    /// <summary>Embeds interleaved RGB8 pixels, row-major.</summary>
    public unsafe Task<float[]> EmbedAsync(ReadOnlyMemory<byte> rgb, int width, int height,
                                    EmbeddingDimension dim = EmbeddingDimension.D768,
                                    CancellationToken ct = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException(
                $"e2: rgb has {rgb.Length} bytes, expected {width}x{height}x3", nameof(rgb));
        int d = Size(dim);

        return GatedAsync(ctx =>
        {
            var buf = new float[d];
            fixed (byte* src = rgb.Span)
            fixed (float* p = buf)
                E2Library.Check(NativeMethods.e2_embed_image_rgb(ctx, src, width, height, p, d),
                                "e2_embed_image_rgb");
            return buf;
        }, ct);
    }

    /// <summary>Embeds mono float32 PCM at 16 kHz, as <c>e2_audio_read</c> returns it.</summary>
    public unsafe Task<float[]> EmbedAsync(ReadOnlyMemory<float> samples16kHzMono,
                                    EmbeddingDimension dim = EmbeddingDimension.D768,
                                    CancellationToken ct = default)
    {
        if (samples16kHzMono.IsEmpty)
            throw new ArgumentException("e2: empty audio buffer", nameof(samples16kHzMono));
        int d = Size(dim);

        return GatedAsync(ctx =>
        {
            var buf = new float[d];
            fixed (float* src = samples16kHzMono.Span)
            fixed (float* p = buf)
                E2Library.Check(NativeMethods.e2_embed_audio(ctx, src, samples16kHzMono.Length, p, d),
                                "e2_embed_audio");
            return buf;
        }, ct);
    }

    /// <summary>Embeds a video given as already-sampled frame files.</summary>
    public unsafe Task<float[]> EmbedVideoAsync(IReadOnlyList<string> framePaths,
                                        EmbeddingDimension dim = EmbeddingDimension.D768,
                                        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(framePaths);
        if (framePaths.Count == 0) throw new ArgumentException("e2: no frames", nameof(framePaths));
        int d = Size(dim);

        return GatedAsync(ctx =>
        {
            using var arena = new UnmanagedArena();
            var ptrs = (IntPtr*)arena.PointerArray(framePaths.Count);
            for (int i = 0; i < framePaths.Count; i++) ptrs[i] = arena.Utf8(framePaths[i]);

            var buf = new float[d];
            fixed (float* p = buf)
                E2Library.Check(NativeMethods.e2_embed_video(ctx, ptrs, framePaths.Count, p, d),
                                "e2_embed_video");
            return buf;
        }, ct);
    }

    /// <summary>
    /// Embeds text with inline <c>&lt;|image|&gt;</c>, <c>&lt;|audio|&gt;</c> and
    /// <c>&lt;|video|&gt;</c> placeholders.
    /// </summary>
    public unsafe Task<float[]> EmbedAsync(E2Input input, EmbeddingDimension dim = EmbeddingDimension.D768,
                                    CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        Validate(input);
        int d = Size(dim);

        return GatedAsync(ctx =>
        {
            using var arena = new UnmanagedArena();

            var native = new NativeMethods.NativeInput
            {
                Text = arena.Utf8(input.Text),
                NImages = input.Images.Count,
                NVideos = input.Videos.Count,
                NAudio = input.Audio.Count,
            };

            if (native.NImages > 0)
            {
                var images = (IntPtr*)arena.PointerArray(native.NImages);
                for (int i = 0; i < native.NImages; i++) images[i] = arena.Utf8(input.Images[i]);
                native.Images = (IntPtr)images;
            }

            if (native.NVideos > 0)
            {
                var videos = (IntPtr*)arena.PointerArray(native.NVideos);
                var frames = new int[native.NVideos];
                for (int i = 0; i < native.NVideos; i++)
                {
                    var paths = input.Videos[i];
                    frames[i] = paths.Count;
                    var list = (IntPtr*)arena.PointerArray(paths.Count);
                    for (int j = 0; j < paths.Count; j++) list[j] = arena.Utf8(paths[j]);
                    videos[i] = (IntPtr)list;
                }
                native.Videos = (IntPtr)videos;
                native.VideoFrames = arena.IntArray(frames);
            }

            if (native.NAudio > 0)
            {
                var audio = (IntPtr*)arena.PointerArray(native.NAudio);
                var lengths = new int[native.NAudio];
                for (int i = 0; i < native.NAudio; i++)
                {
                    lengths[i] = input.Audio[i].Length;
                    audio[i] = arena.FloatArray(input.Audio[i].Span);
                }
                native.Audio = (IntPtr)audio;
                native.AudioLen = arena.IntArray(lengths);
            }

            var buf = new float[d];
            fixed (float* p = buf)
                E2Library.Check(NativeMethods.e2_embed(ctx, &native, p, d), "e2_embed");
            return buf;
        }, ct);
    }

    /* ---------------------------- sync wrappers ----------------------- */

    public float[] Embed(string text, EmbeddingDimension dim = EmbeddingDimension.D768,
                         CancellationToken ct = default)
        => EmbedAsync(text, dim, ct).GetAwaiter().GetResult();

    public float[] Embed(string task, string text, EmbeddingDimension dim = EmbeddingDimension.D768,
                         CancellationToken ct = default)
        => EmbedAsync(task, text, dim, ct).GetAwaiter().GetResult();

    public float[][] EmbedBatch(IReadOnlyList<string> texts,
                                EmbeddingDimension dim = EmbeddingDimension.D768,
                                CancellationToken ct = default)
        => EmbedBatchAsync(texts, dim, ct).GetAwaiter().GetResult();

    public float[] Embed(FileInfo file, EmbeddingDimension dim = EmbeddingDimension.D768,
                         CancellationToken ct = default)
        => EmbedAsync(file, dim, ct).GetAwaiter().GetResult();

    public float[] Embed(ReadOnlyMemory<byte> rgb, int width, int height,
                         EmbeddingDimension dim = EmbeddingDimension.D768,
                         CancellationToken ct = default)
        => EmbedAsync(rgb, width, height, dim, ct).GetAwaiter().GetResult();

    public float[] Embed(ReadOnlyMemory<float> samples16kHzMono,
                         EmbeddingDimension dim = EmbeddingDimension.D768,
                         CancellationToken ct = default)
        => EmbedAsync(samples16kHzMono, dim, ct).GetAwaiter().GetResult();

    public float[] EmbedVideo(IReadOnlyList<string> framePaths,
                              EmbeddingDimension dim = EmbeddingDimension.D768,
                              CancellationToken ct = default)
        => EmbedVideoAsync(framePaths, dim, ct).GetAwaiter().GetResult();

    public float[] Embed(E2Input input, EmbeddingDimension dim = EmbeddingDimension.D768,
                         CancellationToken ct = default)
        => EmbedAsync(input, dim, ct).GetAwaiter().GetResult();

    /* ------------------------------------------------------------------ */

    /* Property getters must not touch the context after Dispose: a closed
     * SafeHandle hands out a stale pointer. */
    private IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _ctx.DangerousGetHandle();
        }
    }

    private static int Size(EmbeddingDimension dim) => dim switch
    {
        EmbeddingDimension.D128 => 128,
        EmbeddingDimension.D256 => 256,
        EmbeddingDimension.D512 => 512,
        EmbeddingDimension.D768 => 768,
        _ => throw new ArgumentOutOfRangeException(nameof(dim), dim, "dim must be 128, 256, 512 or 768"),
    };

    private unsafe delegate void OutCall(IntPtr ctx, float* outv);

    private unsafe Task<float[]> EmbedWith(int dim, OutCall call, CancellationToken ct)
        => GatedAsync(ctx =>
        {
            var buf = new float[dim];
            fixed (float* p = buf) call(ctx, p);
            return buf;
        }, ct);

    /* One forward at a time. The native call is uninterruptible, so `ct` can
     * only act while queued; a running embed always completes. */
    private async Task<T> GatedAsync<T>(Func<IntPtr, T> work, CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ct.ThrowIfCancellationRequested();

        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ct.ThrowIfCancellationRequested();
            /* Queued callers hold no thread; only the running one does. */
            return await Task.Run(() => work(Handle), CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    private void Exclusive(Action<IntPtr> action)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _gate.Wait();
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            action(Handle);
        }
        finally
        {
            _gate.Release();
        }
    }

    private static void Validate(E2Input input)
    {
        int images = Count(input.Text, "<|image|>");
        int videos = Count(input.Text, "<|video|>");
        int audio = Count(input.Text, "<|audio|>");
        if (images != input.Images.Count)
            throw new ArgumentException(
                $"e2: text has {images} <|image|> placeholder(s) but {input.Images.Count} image(s) were given",
                nameof(input));
        if (videos != input.Videos.Count)
            throw new ArgumentException(
                $"e2: text has {videos} <|video|> placeholder(s) but {input.Videos.Count} video(s) were given",
                nameof(input));
        if (audio != input.Audio.Count)
            throw new ArgumentException(
                $"e2: text has {audio} <|audio|> placeholder(s) but {input.Audio.Count} audio buffer(s) were given",
                nameof(input));
        if (input.Videos.Any(v => v.Count == 0))
            throw new ArgumentException("e2: a video needs at least one frame", nameof(input));
        if (input.Audio.Any(a => a.IsEmpty))
            throw new ArgumentException("e2: empty audio buffer", nameof(input));
    }

    private static int Count(string text, string needle)
    {
        int n = 0;
        for (int i = text.IndexOf(needle, StringComparison.Ordinal); i >= 0;
             i = text.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
            n++;
        return n;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        /* Do not free the context under a running call. The gate itself is
         * never disposed: a queued caller then observes ObjectDisposedException
         * instead of ObjectDisposedException from a disposed semaphore. */
        _gate.Wait();
        try
        {
            _ctx.Dispose();
        }
        finally
        {
            _gate.Release();
        }
    }

    private sealed class ContextHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        internal ContextHandle(IntPtr handle) : base(ownsHandle: true) => SetHandle(handle);

        protected override bool ReleaseHandle()
        {
            NativeMethods.e2_free(handle);
            return true;
        }
    }
}
