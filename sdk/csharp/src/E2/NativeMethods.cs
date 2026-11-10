using System.Runtime.InteropServices;

namespace E2;

/* 1:1 with ../../../../e2.h. Nothing here is public API; E2Model is. */
internal static unsafe partial class NativeMethods
{
    internal const string Lib = "e2";

    /* Must match E2_API_VERSION in e2.h. Checked at load, see E2Library. */
    internal const int ApiVersion = 1;

    internal const int DevText = 1, DevImage = 2, DevAudio = 4, DevVideo = 8;
    internal const int DeviceCpu = 0, DeviceGpu = 1;

    [LibraryImport(Lib)]
    internal static partial int e2_api_version();

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial IntPtr e2_load(string modelPath, string tokenizerPath);

    [LibraryImport(Lib)]
    internal static partial void e2_free(IntPtr ctx);

    [LibraryImport(Lib)]
    internal static partial int e2_modalities(IntPtr ctx);

    [LibraryImport(Lib)]
    internal static partial void e2_set_threads(IntPtr ctx, int nthreads);

    [LibraryImport(Lib)]
    internal static partial int e2_threads(IntPtr ctx);

    [LibraryImport(Lib)]
    internal static partial int e2_set_device(IntPtr ctx, int device);

    [LibraryImport(Lib)]
    internal static partial int e2_device(IntPtr ctx);

    [LibraryImport(Lib)]
    internal static partial IntPtr e2_error();

    [LibraryImport(Lib)]
    internal static partial IntPtr e2_model_info(IntPtr ctx);

    [LibraryImport(Lib)]
    internal static partial double e2_last_ms(IntPtr ctx);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int e2_embed_text(IntPtr ctx, string text, float* outv, int dim);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int e2_embed_text_task(IntPtr ctx, string task, string text,
                                                   float* outv, int dim);

    [LibraryImport(Lib)]
    internal static partial int e2_embed_text_batch(IntPtr ctx, IntPtr* texts, int n,
                                                    float* outv, int dim);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int e2_embed_image(IntPtr ctx, string path, float* outv, int dim);

    [LibraryImport(Lib)]
    internal static partial int e2_embed_image_rgb(IntPtr ctx, byte* rgb, int width, int height,
                                                   float* outv, int dim);

    [LibraryImport(Lib)]
    internal static partial int e2_embed_audio(IntPtr ctx, float* samples, int nsamples,
                                               float* outv, int dim);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int e2_audio_read(string path, out IntPtr samples, out int nsamples);

    [LibraryImport(Lib)]
    internal static partial void e2_free_buffer(IntPtr p);

    [LibraryImport(Lib)]
    internal static partial int e2_embed_video(IntPtr ctx, IntPtr* framePaths, int nframes,
                                               float* outv, int dim);

    [LibraryImport(Lib)]
    internal static partial int e2_embed(IntPtr ctx, NativeInput* input, float* outv, int dim);

    /* Mirror of e2_input. Pointer fields are unmanaged blocks owned by the
     * caller for the duration of the call. */
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeInput
    {
        public IntPtr Text;
        public IntPtr Images;       /* char**  */
        public int NImages;
        public IntPtr Videos;       /* char*** */
        public int NVideos;
        public IntPtr VideoFrames;  /* int*    */
        public IntPtr Audio;        /* float** */
        public IntPtr AudioLen;     /* int*    */
        public int NAudio;
    }
}
