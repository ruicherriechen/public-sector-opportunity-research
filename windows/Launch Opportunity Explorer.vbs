Option Explicit

' Resolve every path from this script, not from the shortcut's working folder.
Dim fileSystem, scriptDirectory, projectRoot, pythonwPath, launcherPath
Dim shell, command
Set fileSystem = CreateObject("Scripting.FileSystemObject")
scriptDirectory = fileSystem.GetParentFolderName(WScript.ScriptFullName)
projectRoot = fileSystem.GetParentFolderName(scriptDirectory)
pythonwPath = fileSystem.BuildPath(projectRoot, ".venv\Scripts\pythonw.exe")
launcherPath = fileSystem.BuildPath(projectRoot, "src\public_sector_opportunity_research\windows_launcher.py")

If Not fileSystem.FileExists(pythonwPath) Then
    MsgBox "The project's Python environment is missing. Recreate .venv and install the requirements before launching the application.", vbCritical, "Public Sector Opportunity Explorer"
    WScript.Quit 1
End If

If Not fileSystem.FileExists(launcherPath) Then
    MsgBox "The Opportunity Explorer launcher file is missing from the project.", vbCritical, "Public Sector Opportunity Explorer"
    WScript.Quit 1
End If

Set shell = CreateObject("WScript.Shell")
command = Quote(pythonwPath) & " " & Quote(launcherPath) & " start --project-root " & Quote(projectRoot)
' Window style 0 keeps normal launches quiet; Python displays a message box on errors.
shell.Run command, 0, False

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
