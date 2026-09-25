#ifndef AppVersion
#define AppVersion "0.1.0"
#endif

[Setup]
AppId={{E7C3A91D-4B62-4F0A-9C55-6D8E2F1A0B73}
AppName=h3
AppVersion={#AppVersion}
AppPublisher=stepupgaming
DefaultDirName={localappdata}\Programs\h3
DefaultGroupName=h3
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=h3-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
ChangesEnvironment=yes
WizardStyle=modern
UninstallDisplayName=h3
InfoBeforeFile=info.txt
InfoAfterFile=after.txt
CloseApplications=no

[Files]
Source: "..\target\release\h3.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\comfy-workflows\*"; DestDir: "{app}\comfy-workflows"; Flags: ignoreversion recursesubdirs; Excludes: "node_modules\*,*\node_modules\*,.git\*"
Source: "..\runtimes\minimax-h3\*"; DestDir: "{app}\runtimes\minimax-h3"; Flags: ignoreversion recursesubdirs; Excludes: ".venv\*,*\.venv\*,__pycache__\*,*\__pycache__\*,*.pyc,.git\*,ComfyUI\extra_model_paths.yaml,ComfyUI\models\*,ComfyUI\input\*,ComfyUI\output\*,ComfyUI\temp\*,ComfyUI\user\*,ComfyUI\custom_nodes\ComfyUI-NVIDIA-DLSS-Frame-Interpolation\bin\runtime\*,ComfyUI\custom_nodes\ComfyUI-NVIDIA-DLSS-Frame-Interpolation\Assets\*.mp4"

[Tasks]
Name: addtopath; Description: "Add h3 to the user PATH"; GroupDescription: "Options:"; Flags: checkedonce

[Code]
const
  EnvironmentKey = 'Environment';

function DeletePathSegment(const Paths, Segment: string): string;
var
  Rest: string;
  Item: string;
  Joined: string;
  Semi: Integer;
begin
  Rest := Paths;
  Joined := '';
  while Rest <> '' do
  begin
    Semi := Pos(';', Rest);
    if Semi = 0 then
    begin
      Item := Rest;
      Rest := '';
    end
    else
    begin
      Item := Copy(Rest, 1, Semi - 1);
      Rest := Copy(Rest, Semi + 1, Length(Rest));
    end;
    if (Item <> '') and (CompareText(Item, Segment) <> 0) then
    begin
      if Joined <> '' then
        Joined := Joined + ';';
      Joined := Joined + Item;
    end;
  end;
  Result := Joined;
end;

procedure WriteUserPath(const Paths: string);
begin
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths);
end;

procedure AddInstallDirToPath();
var
  Paths, AppPath: string;
begin
  AppPath := ExpandConstant('{app}');
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(AppPath) + ';', ';' + Uppercase(Paths) + ';') = 0 then
  begin
    if Paths = '' then
      Paths := AppPath
    else
      Paths := Paths + ';' + AppPath;
    WriteUserPath(Paths);
  end;
end;

procedure RemoveInstallDirFromPath();
var
  Paths, AppPath: string;
begin
  AppPath := ExpandConstant('{app}');
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Paths) then
    exit;
  WriteUserPath(DeletePathSegment(Paths, AppPath));
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddInstallDirToPath();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveInstallDirFromPath();
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtimes\minimax-h3\.venv"
Type: filesandordirs; Name: "{app}\runtimes\minimax-h3\jev-sdk\.venv"
Type: filesandordirs; Name: "{app}\runtimes\minimax-h3\ComfyUI\input"
Type: filesandordirs; Name: "{app}\runtimes\minimax-h3\ComfyUI\output"
Type: filesandordirs; Name: "{app}\runtimes\minimax-h3\ComfyUI\temp"
Type: filesandordirs; Name: "{app}\runtimes\minimax-h3\ComfyUI\user"
Type: files; Name: "{app}\runtimes\minimax-h3\ComfyUI\extra_model_paths.yaml"
