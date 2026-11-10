using System.Globalization;
using Xunit;

namespace E2.Tests;

internal static class Repo
{
    public static string Root { get; } = FindRoot();

    private static string FindRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (; dir is not null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "e2.h")))
                return dir.FullName;
        throw new InvalidOperationException($"no repo root (e2.h) above {AppContext.BaseDirectory}");
    }

    public static string ModelPath =>
        Environment.GetEnvironmentVariable("E2_MODEL")
        ?? Path.Combine(Root, "models", "e2-text+image+audio.bin");

    public static string TokenizerPath =>
        Environment.GetEnvironmentVariable("E2_TOKENIZER")
        ?? Path.Combine(Root, "models", "e2-tokenizer.bin");
}

/* One loaded model for the whole suite: loading is ~1 s and each context maps
 * 800 MB (shared page cache, but still). Null when models/ is absent. */
public sealed class ModelFixture : IDisposable
{
    public E2Model? Model { get; }

    public ModelFixture()
    {
        if (!File.Exists(Repo.ModelPath) || !File.Exists(Repo.TokenizerPath))
        {
            Console.WriteLine($"skipping: no model at {Repo.ModelPath}");
            return;
        }
        Model = E2Model.Load(Repo.ModelPath, Repo.TokenizerPath);
    }

    public void Dispose() => Model?.Dispose();
}

[CollectionDefinition("model")]
public sealed class ModelCollection : ICollectionFixture<ModelFixture>;

internal static class Vec
{
    public static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    public static float[] Normalized(ReadOnlySpan<float> v)
    {
        double n = 0;
        foreach (var x in v) n += x * x;
        n = Math.Sqrt(n);
        var r = new float[v.Length];
        for (int i = 0; i < v.Length; i++) r[i] = (float)(v[i] / n);
        return r;
    }

    public static double Norm(float[] a)
    {
        double n = 0;
        foreach (var x in a) n += x * x;
        return Math.Sqrt(n);
    }
}

/* golden/embeddings.txt, written by tools/e2_golden.c: "<name> <dim>" then the
 * dim values. Names can contain spaces, so dim is the last token. */
internal static class Golden
{
    private static readonly Lazy<Dictionary<string, float[]>> s_entries = new(Load);

    public static IReadOnlyDictionary<string, float[]> Entries => s_entries.Value;

    private static Dictionary<string, float[]> Load()
    {
        var map = new Dictionary<string, float[]>(StringComparer.Ordinal);
        var path = Path.Combine(AppContext.BaseDirectory, "golden", "embeddings.txt");
        if (!File.Exists(path))
        {
            Console.WriteLine($"skipping: no golden file at {path} (`make golden`)");
            return map;
        }

        var lines = File.ReadAllLines(path);
        for (int i = 0; i + 1 < lines.Length; i += 2)
        {
            int split = lines[i].LastIndexOf(' ');
            var name = lines[i][..split];
            var dim = int.Parse(lines[i][(split + 1)..], CultureInfo.InvariantCulture);
            var values = lines[i + 1]
                .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => float.Parse(s, CultureInfo.InvariantCulture))
                .ToArray();
            Assert.Equal(dim, values.Length);
            map[name] = values;
        }
        return map;
    }
}
