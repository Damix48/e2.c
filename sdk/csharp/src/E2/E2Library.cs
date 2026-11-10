using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace E2;

/* Finds libe2 next to the app, in runtimes/<rid>/native (nupkg layout), or
 * wherever E2_NATIVE_LIB points. */
internal static class E2Library
{
    private static bool s_checked;

    [ModuleInitializer]
    internal static void Init()
    {
        NativeLibrary.SetDllImportResolver(typeof(E2Library).Assembly, Resolve);
    }

    internal static string FileName =>
        OperatingSystem.IsMacOS() ? "libe2.dylib" : "libe2.so";

    /* Called once by E2Model.Load. Turns a missing/stale library into an
     * E2Exception that says what to do. */
    internal static void EnsureAvailable()
    {
        if (s_checked) return;
        int version;
        try
        {
            version = NativeMethods.e2_api_version();
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            throw new E2Exception(
                $"could not load {FileName}. Build it with `make shared` and set " +
                "E2_NATIVE_LIB to its path, or install the E2 package with native assets.", e);
        }
        if (version != NativeMethods.ApiVersion)
            throw new E2Exception(
                $"libe2 ABI version {version} does not match this assembly ({NativeMethods.ApiVersion}). " +
                "Rebuild libe2 from the same source tree as the binding.");
        s_checked = true;
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? path)
    {
        if (name != NativeMethods.Lib) return IntPtr.Zero;
        foreach (var candidate in Candidates())
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle))
                return handle;
        return IntPtr.Zero; /* let the default search try a system-wide install */
    }

    private static IEnumerable<string> Candidates()
    {
        var env = Environment.GetEnvironmentVariable("E2_NATIVE_LIB");
        if (!string.IsNullOrEmpty(env)) yield return env;

        var rid = RuntimeInformation.RuntimeIdentifier;
        var dir = AppContext.BaseDirectory;
        /* Walk up: `dotnet test` from sdk/csharp must find the repo-root build. */
        for (int i = 0; i < 8 && !string.IsNullOrEmpty(dir); i++)
        {
            yield return Path.Combine(dir, FileName);
            yield return Path.Combine(dir, "runtimes", rid, "native", FileName);
            yield return Path.Combine(dir, "src", "E2", "runtimes", rid, "native", FileName);
            dir = Path.GetDirectoryName(dir.TrimEnd(Path.DirectorySeparatorChar));
        }
    }

    /* e2_error() is thread-local and invalidated by the next failure, so the
     * message is copied on this line. */
    internal static void Check(int rc, string call)
    {
        if (rc != 0)
            throw new E2Exception($"{call}: {Marshal.PtrToStringUTF8(NativeMethods.e2_error())}");
    }
}
