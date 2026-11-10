using System.Diagnostics;
using E2;

/* Throughput/latency of the C# binding, including the interop overhead.
 * The engine's own LastMilliseconds is printed next to the wall time, so the
 * cost of the marshalling layer is visible.
 *
 * usage: dotnet run -c Release --project sdk/csharp/tests/E2.Bench -- [--iters N] [--threads N] */

int iters = 10;
int threads = 0; /* 0 = engine default */
string? modelPath = null;
string? tokenizerPath = null;

for (int i = 0; i < args.Length - 1; i++)
    switch (args[i])
    {
        case "--iters": iters = int.Parse(args[++i]); break;
        case "--threads": threads = int.Parse(args[++i]); break;
        case "--model": modelPath = args[++i]; break;
        case "--tokenizer": tokenizerPath = args[++i]; break;
    }

var root = new DirectoryInfo(AppContext.BaseDirectory);
while (root is not null && !File.Exists(Path.Combine(root.FullName, "e2.h"))) root = root.Parent;
if (root is null) { Console.Error.WriteLine("e2.h not found above the binary"); return 2; }

modelPath ??= Path.Combine(root.FullName, "models", "e2-text+image+audio.bin");
tokenizerPath ??= Path.Combine(root.FullName, "models", "e2-tokenizer.bin");
if (!File.Exists(modelPath)) { Console.Error.WriteLine($"no model at {modelPath}"); return 2; }

using var model = E2Model.Load(modelPath, tokenizerPath);
if (threads > 0) model.Threads = threads;

Console.WriteLine($"model    {Path.GetFileName(modelPath)}");
Console.WriteLine($"engine   {model.ModelInfo}");
Console.WriteLine($"threads  {model.Threads}   iters {iters}");
Console.WriteLine();
Console.WriteLine($"{"scenario",-32} {"n",4} {"wall ms",9} {"engine ms",10} {"per s",9}");

string[] texts =
[
    "hello world",
    "a brown fox jumps over the lazy dog",
    "the quick brown fox jum",
    "embedding models map text to vectors",
    "one",
    "what is the capital of Italy",
    "short query",
    "a slightly longer document about retrieval augmented generation and embeddings",
];
var batch = Enumerable.Range(0, 32).Select(i => texts[i % texts.Length]).ToArray();
var rgb = Rgb(224, 224);
var pcm = Pcm(64000);
var wav = WriteWav(Path.GetTempPath(), Pcm(48000 * 30)); /* 30 s, the truncation limit */

void Row(string name, int n, double wallMs, double engineMs, double units)
{
    string per = units <= 0 ? "" : $"{units / (wallMs / 1000.0):F1}";
    Console.WriteLine($"{name,-32} {n,4} {wallMs,9:F1} {engineMs,10:F1} {per,9}");
}

double Time(int n, Action action)
{
    action(); /* warmup */
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < n; i++) action();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / n;
}

string text = texts[0];
double ms = Time(iters, () => model.Embed(text));
Row("text 768 (sync)", iters, ms, model.LastMilliseconds, 1);

ms = Time(iters, () => model.Embed(text, EmbeddingDimension.D128));
Row("text 128 (sync)", iters, ms, model.LastMilliseconds, 1);

ms = Time(Math.Max(iters / 2, 1), () => model.EmbedBatch(batch));
Row("text batch 32", 32, ms, model.LastMilliseconds, 32);

await model.EmbedAsync(text); /* warmup */
var swAsync = Stopwatch.StartNew();
for (int i = 0; i < iters; i++) await model.EmbedAsync(text);
swAsync.Stop();
Row("text 768 (async, 1 inflight)", iters, swAsync.Elapsed.TotalMilliseconds / iters,
    model.LastMilliseconds, 1);

if ((model.Modalities & E2Model.Modality.Image) != 0)
{
    ms = Time(iters, () => model.Embed(rgb, 224, 224));
    Row("image 224x224 rgb", iters, ms, model.LastMilliseconds, 1);
}

if ((model.Modalities & E2Model.Modality.Audio) != 0)
{
    int n = Math.Max(iters / 2, 1);
    ms = Time(n, () => model.Embed(pcm));
    Row("audio 4 s samples", n, ms, model.LastMilliseconds, 1);

    int m = Math.Max(iters / 4, 1);
    ms = Time(m, () => model.Embed(new FileInfo(wav)));
    Row("audio 30 s file (wav)", m, ms, model.LastMilliseconds, 1);
}

/* Queued on one model: the engine runs one forward at a time, so this is the
 * gate plus the engine, not parallel speedup. */
double Burst(int n, int count, Action action)
{
    void Round()
    {
        var tasks = new Task[count];
        for (int i = 0; i < count; i++) tasks[i] = Task.Run(action);
        Task.WaitAll(tasks);
    }

    Round();
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < n; i++) Round();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / n;
}

ms = Burst(iters, 8, () => model.Embed(text));
Row("text 8 queued (1 model)", iters * 8, ms, model.LastMilliseconds, 8);

/* One model per worker: the intended shape for concurrent requests. Weights
 * are read-only mmap, so the extra contexts share the page cache. */
int workers = Math.Min(4, Math.Max(2, Environment.ProcessorCount / 4));
var pool = Enumerable.Range(0, workers)
                     .Select(_ => E2Model.Load(modelPath, tokenizerPath))
                     .ToArray();
foreach (var m in pool) m.Threads = Math.Max(1, Environment.ProcessorCount / pool.Length);
try
{
    int next = -1;
    ms = Burst(Math.Max(iters / 2, 1), pool.Length, () =>
    {
        var m = pool[(int)((uint)Interlocked.Increment(ref next) % (uint)pool.Length)];
        m.Embed(text);
    });
    Row($"text {pool.Length} parallel (N models)", Math.Max(iters / 2, 1) * pool.Length,
        ms, pool[0].LastMilliseconds, pool.Length);
}
finally
{
    foreach (var m in pool) m.Dispose();
    File.Delete(wav);
}

return 0;

static byte[] Rgb(int w, int h)
{
    var rgb = new byte[w * h * 3];
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int o = (y * w + x) * 3;
            rgb[o] = (byte)((x * 7 + y * 13) & 255);
            rgb[o + 1] = (byte)((x * 11 + y * 3) & 255);
            rgb[o + 2] = (byte)((x * 5 + y * 17) & 255);
        }
    return rgb;
}

static float[] Pcm(int n)
{
    var pcm = new float[n];
    for (int i = 0; i < n; i++) pcm[i] = ((i % 200) - 100) / 512f;
    return pcm;
}

static string WriteWav(string dir, ReadOnlySpan<float> mono, int sampleRate = 16000)
{
    var path = Path.Combine(dir, "e2-bench.wav");
    using var w = new BinaryWriter(File.Create(path));
    w.Write("RIFF"u8); w.Write(36 + mono.Length * 2); w.Write("WAVE"u8);
    w.Write("fmt "u8); w.Write(16); w.Write((short)1); w.Write((short)1);
    w.Write(sampleRate); w.Write(sampleRate * 2); w.Write((short)2); w.Write((short)16);
    w.Write("data"u8); w.Write(mono.Length * 2);
    foreach (var s in mono) w.Write((short)Math.Clamp((int)MathF.Round(s * 32767f), -32768, 32767));
    return path;
}
