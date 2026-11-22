<#
.SYNOPSIS
Creates per-user desktop shortcuts for starting and stopping Opportunity Explorer.

.DESCRIPTION
This setup script changes no execution policy, installs nothing system-wide,
and requires no administrator privileges. The normal shortcuts invoke Windows
Script Host, which starts the existing .venv Python process without a console.
#>
[CmdletBinding()]
param(
    [string]$DestinationDirectory = [Environment]::GetFolderPath("DesktopDirectory")
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$startScript = Join-Path $PSScriptRoot "Launch Opportunity Explorer.vbs"
$stopScript = Join-Path $PSScriptRoot "Stop Opportunity Explorer.vbs"
$wscriptPath = Join-Path $env:SystemRoot "System32\wscript.exe"
$iconPath = Join-Path $env:SystemRoot "System32\imageres.dll"

foreach ($requiredPath in @($startScript, $stopScript, $wscriptPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required launcher file is missing: $requiredPath"
    }
}
if (-not (Test-Path -LiteralPath $DestinationDirectory -PathType Container)) {
    throw "Desktop folder could not be found: $DestinationDirectory"
}

$shell = New-Object -ComObject WScript.Shell

function New-ExplorerShortcut {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$ScriptPath,
        [Parameter(Mandatory)] [string]$Description,
        [Parameter(Mandatory)] [int]$IconIndex
    )

    $shortcutPath = Join-Path $DestinationDirectory "$Name.lnk"
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $wscriptPath
    $shortcut.Arguments = '"' + $ScriptPath + '"'
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.Description = $Description
    $shortcut.IconLocation = "$iconPath,$IconIndex"
    $shortcut.Save()
    return $shortcutPath
}

$startShortcut = New-ExplorerShortcut `
    -Name "Public Sector Opportunity Explorer" `
    -ScriptPath $startScript `
    -Description "Open the local Public Sector Opportunity Explorer" `
    -IconIndex 14
$stopShortcut = New-ExplorerShortcut `
    -Name "Stop Opportunity Explorer" `
    -ScriptPath $stopScript `
    -Description "Safely stop the local Public Sector Opportunity Explorer" `
    -IconIndex 99

Write-Host "Created desktop shortcuts:"
Write-Host "  $startShortcut"
Write-Host "  $stopShortcut"
