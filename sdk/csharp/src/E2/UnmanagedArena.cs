using System.Runtime.InteropServices;
using System.Text;

namespace E2;

/* Owns the unmanaged blocks that back one native call (UTF-8 strings and
 * pointer/int/float arrays). Freed on Dispose, i.e. after the call returns. */
internal sealed unsafe class UnmanagedArena : IDisposable
{
    private readonly List<IntPtr> _blocks = [];

    private void* Alloc(nuint bytes)
    {
        var p = (IntPtr)NativeMemory.AllocZeroed(bytes);
        _blocks.Add(p);
        return (void*)p;
    }

    internal IntPtr Utf8(string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        var p = (byte*)Alloc((nuint)bytes.Length + 1);
        bytes.AsSpan().CopyTo(new Span<byte>(p, bytes.Length));
        p[bytes.Length] = 0;
        return (IntPtr)p;
    }

    internal IntPtr PointerArray(int count) => (IntPtr)Alloc((nuint)(sizeof(IntPtr) * Math.Max(count, 1)));

    internal IntPtr IntArray(ReadOnlySpan<int> values)
    {
        var p = (int*)Alloc((nuint)(sizeof(int) * Math.Max(values.Length, 1)));
        values.CopyTo(new Span<int>(p, values.Length));
        return (IntPtr)p;
    }

    internal IntPtr FloatArray(ReadOnlySpan<float> values)
    {
        var p = (float*)Alloc((nuint)(sizeof(float) * Math.Max(values.Length, 1)));
        values.CopyTo(new Span<float>(p, values.Length));
        return (IntPtr)p;
    }

    public void Dispose()
    {
        foreach (var p in _blocks) NativeMemory.Free((void*)p);
        _blocks.Clear();
    }
}
