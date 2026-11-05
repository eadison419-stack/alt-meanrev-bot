$credentialStoreSource = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class AltMeanrevCredentialStore
{
    private const uint CredentialTypeGeneric = 1;
    private const uint CredentialPersistLocalMachine = 2;

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct NativeCredential
    {
        public uint Flags;
        public uint Type;
        [MarshalAs(UnmanagedType.LPWStr)] public string TargetName;
        [MarshalAs(UnmanagedType.LPWStr)] public string Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public uint CredentialBlobSize;
        public IntPtr CredentialBlob;
        public uint Persist;
        public uint AttributeCount;
        public IntPtr Attributes;
        [MarshalAs(UnmanagedType.LPWStr)] public string TargetAlias;
        [MarshalAs(UnmanagedType.LPWStr)] public string UserName;
    }

    [DllImport("Advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredWrite(ref NativeCredential credential, uint flags);

    [DllImport("Advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CredRead(string target, uint type, uint flags, out IntPtr credential);

    [DllImport("Advapi32.dll", EntryPoint = "CredFree", SetLastError = false)]
    private static extern void CredFree(IntPtr credential);

    public static void Write(string target, string userName, string secret)
    {
        if (String.IsNullOrWhiteSpace(target) || String.IsNullOrWhiteSpace(userName) || String.IsNullOrWhiteSpace(secret))
            throw new ArgumentException("Credential fields must not be empty.");

        byte[] blob = System.Text.Encoding.Unicode.GetBytes(secret);
        if (blob.Length > 2560)
        {
            Array.Clear(blob, 0, blob.Length);
            throw new ArgumentOutOfRangeException("secret", "The credential secret exceeds the Windows Credential Manager limit.");
        }

        IntPtr blobPointer = IntPtr.Zero;
        try
        {
            blobPointer = Marshal.AllocHGlobal(blob.Length);
            Marshal.Copy(blob, 0, blobPointer, blob.Length);
            NativeCredential credential = new NativeCredential();
            credential.Type = CredentialTypeGeneric;
            credential.TargetName = target;
            credential.UserName = userName;
            credential.CredentialBlobSize = (uint)blob.Length;
            credential.CredentialBlob = blobPointer;
            credential.Persist = CredentialPersistLocalMachine;
            if (!CredWrite(ref credential, 0))
                throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        finally
        {
            if (blobPointer != IntPtr.Zero)
            {
                for (int i = 0; i < blob.Length; i++) Marshal.WriteByte(blobPointer, i, 0);
                Marshal.FreeHGlobal(blobPointer);
            }
            Array.Clear(blob, 0, blob.Length);
        }
    }

    public static string[] Read(string target)
    {
        IntPtr credentialPointer;
        if (!CredRead(target, CredentialTypeGeneric, 0, out credentialPointer))
            throw new Win32Exception(Marshal.GetLastWin32Error());

        byte[] blob = null;
        try
        {
            NativeCredential credential = (NativeCredential)Marshal.PtrToStructure(credentialPointer, typeof(NativeCredential));
            string userName = credential.UserName;
            blob = new byte[credential.CredentialBlobSize];
            if (blob.Length > 0) Marshal.Copy(credential.CredentialBlob, blob, 0, blob.Length);
            string secret = System.Text.Encoding.Unicode.GetString(blob);
            return new string[] { userName, secret };
        }
        finally
        {
            if (blob != null) Array.Clear(blob, 0, blob.Length);
            CredFree(credentialPointer);
        }
    }
}
'@

if (-not ('AltMeanrevCredentialStore' -as [type])) {
    Add-Type -TypeDefinition $credentialStoreSource -Language CSharp -ErrorAction Stop
}
