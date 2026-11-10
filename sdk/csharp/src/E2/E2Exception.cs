namespace E2;

/// <summary>The native engine reported an error, or could not be loaded.</summary>
public sealed class E2Exception : Exception
{
    public E2Exception(string message) : base(message) { }

    public E2Exception(string message, Exception inner) : base(message, inner) { }
}
