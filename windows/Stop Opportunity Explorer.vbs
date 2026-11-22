Option Explicit

' The stop entry point uses the same project-relative resolution as startup.
Dim fileSystem, scriptDirectory, projectRoot, pythonwPath, launcherPath
Dim shell, command
Set fileSystem = CreateObject("Scripting.FileSystemObject")
scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
projectRoot = fileSystem.GetParentFolderName(scriptDirectory)
pythonwPath = fileSystem.BuildPath(projectRoot, ".venv\Scripts\pythonw.exe")
launcherPath = fileSystem.BuildPath(projectRoot, "src\public_sector_opportunity_research\windows_launcher.py")

If Not fileSystem.FileExists(pythonwPath) Or Not fileSystem.FileExists(launcherPath) Then
    MsgBox "The Opportunity Explorer project or its Python environment could not be found.", vbCritical, "Stop Opportunity Explorer"
    WScript.Quit 1
End If

Set shell = CreateObject("WScript.Shell")
command = Quote(pythonwPath) & " " & Quote(launcherPath) & " stop --project-root " & Quote(projectRoot)
shell.Run command, 0, False

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
