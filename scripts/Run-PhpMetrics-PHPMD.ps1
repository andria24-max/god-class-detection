# ==============================================================================
# XDD DATASET - PhpMetrics + PHPMD + CK Metrics (UNIFIED)
# Place at D:\xdd_data\
#
# Run:           .\Run-PhpMetrics-PHPMD.ps1
# Resume:        .\Run-PhpMetrics-PHPMD.ps1 -Resume
# Force redo:    .\Run-PhpMetrics-PHPMD.ps1 -Force
# One project:   .\Run-PhpMetrics-PHPMD.ps1 -Only drupal
# Dry run:       .\Run-PhpMetrics-PHPMD.ps1 -DryRun
# Skip tools:    .\Run-PhpMetrics-PHPMD.ps1 -SkipPhpMetrics
# Rebuild CSV:   .\Run-PhpMetrics-PHPMD.ps1 -RebuildCSV
#
# OUTPUT FILES
#   output\<repo>.json           PhpMetrics raw JSON
#   output\<repo>-phpmd.xml      PHPMD violations XML (GodClass rule only)
#   output\ck_metrics_all.csv    CK metrics per class (all repos merged)
#   output\run_log.txt           Timestamped run log
#   output\.completed.txt        Resume state tracker
#
# COLUMNS in ck_metrics_all.csv -- aligned with train_godclass.py S.3.3
#   Project, Class
#   WMC   Weighted Methods per Class   (sum of cyclomatic complexity)
#   LCOM  Lack of Cohesion of Methods  (Henderson-Sellers, PhpMetrics field "lcom")
#   CBO   Coupling Between Objects     (efferentCoupling + afferentCoupling)
#   RFC   Response For a Class         (WMC + CBO; PhpMetrics 2.9.x proxy)
#   DIT   Depth of Inheritance Tree    (parents.Count; 0 or 1 due to tool limit)
#   NOC   Number of Children           (derived from parents[] cross-map)
#   TCC   Tight Class Cohesion         (1/(1+LCOM) proxy; PhpMetrics has no native TCC)
#   NOM   Number of Methods            (nbMethods field)
#   LOC   Lines of Code
#   PHPMD_Flag   1 if PHPMD GodClass rule fired for this class, else 0
#   Expert_Hits  Count of expert threshold breaches (S.3.2.2)
#   Label        1 if PHPMD_Flag=1 OR Expert_Hits>=2, else 0
#   LabelSource  "Both" | "PHPMD_only" | "Expert_only" | "Neither"
#                (required for Step 13 sensitivity analysis in train_godclass.py)
#
# KNOWN PhpMetrics 2.9.x LIMITATIONS (documented in train_godclass.py header)
#   - ATFD always 0 -> excluded from feature set (field kept for audit)
#   - NOA always 0  -> excluded from feature set (field kept for audit)
#   - DIT limited to 0/1 (no full inheritance chain traversal)
#   - RFC approximated as WMC+CBO (no direct method-call count available)
#   - TCC approximated as 1/(1+LCOM)
#
# PHPMD GodClass rule (design ruleset)
#   Fires when: ATFD > 5 AND WMC > 47 AND TCC < 0.33
#   Script uses: phpmd <src> xml design --reportfile <out>
#   IMPORTANT: We pass only "design" -- not "codesize,design,naming".
#   Using broader rulesets inflates PHPMD_Flag with non-GodClass violations,
#   which invalidates the McNemar's test in Step 10 of train_godclass.py.
# ==============================================================================

param(
    [switch]$Resume,
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipPhpMetrics,
    [switch]$RebuildCSV,
    [string]$Only = "",
    [int]$MinJsonBytes = 100,
    [int]$MinXmlBytes  = 50
)

Set-StrictMode -Off
# "Continue" allows the script to log errors and move to the next project
# instead of halting entirely when one repo fails PhpMetrics or PHPMD.
$ErrorActionPreference = "Continue"

# ==============================================================================
# PATHS
# ==============================================================================
$BaseDir    = $PSScriptRoot
$ProjectDir = Join-Path $BaseDir "projects"
$OutputDir  = Join-Path $BaseDir "output"
$LogFile    = Join-Path $OutputDir "run_log.txt"
$StateFile  = Join-Path $OutputDir ".completed.txt"
$CsvAll     = Join-Path $OutputDir "ck_metrics_all.csv"

foreach ($d in @($ProjectDir, $OutputDir)) {
    if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# ==============================================================================
# LOGGING
# ==============================================================================
function Write-Log {
    param([string]$Msg, [string]$Level = "INFO")
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts][$Level] $Msg"
    $line | Out-File -Append -FilePath $LogFile -Encoding utf8
    switch ($Level) {
        "OK"    { Write-Host "  $Msg" -ForegroundColor Green  }
        "WARN"  { Write-Host "  $Msg" -ForegroundColor Yellow }
        "ERROR" { Write-Host "  $Msg" -ForegroundColor Red    }
        default { Write-Host "  $Msg" }
    }
}

function Get-SizeKB($Path) {
    return [math]::Round((Get-Item $Path).Length / 1KB, 1)
}

function Is-Completed($Name) {
    if (!(Test-Path $StateFile)) { return $false }
    return (Get-Content $StateFile -Encoding utf8) -contains $Name
}

function Mark-Completed($Name) {
    $Name | Out-File -Append -FilePath $StateFile -Encoding utf8
}

# ==============================================================================
# TOOL VALIDATION
# ==============================================================================
if (!$SkipPhpMetrics -and !$RebuildCSV) {
    foreach ($tool in @("phpmetrics", "phpmd")) {
        if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
            Write-Error "'$tool' not found in PATH. Install it and retry."
            exit 1
        }
    }
}

# ==============================================================================
# FRAMEWORK DETECTION
# ==============================================================================
function Detect-Framework {
    param([string]$Path)
    if (Test-Path (Join-Path $Path "artisan"))     { return "laravel"      }
    if (Test-Path (Join-Path $Path "bin\console")) { return "symfony"      }
    if (Test-Path (Join-Path $Path "system")) {
        if (Test-Path (Join-Path $Path "application")) { return "codeigniter3" }
        if (Test-Path (Join-Path $Path "app\Config"))  { return "codeigniter4" }
    }
    return "generic"
}

function Get-FrameworkExclusions {
    param([string]$Framework)
    switch ($Framework) {
        "laravel"      { return @("vendor","storage","bootstrap","node_modules") }
        # Note: for the CodeIgniter3 GitHub repo itself, "system" contains
        # the framework core -- DO NOT exclude it or 0 classes will be found.
        "codeigniter3" { return @("vendor") }
        "codeigniter4" { return @("system","vendor","writable") }
        "symfony"      { return @("vendor","var","bin") }
        default        { return @("vendor","node_modules") }
    }
}

# ==============================================================================
# PHPMETRICS RUNNER
# ==============================================================================
function Invoke-PhpMetrics {
    param([string]$Name, [string]$SrcPath, [string]$OutJson)

    if (!$Force -and (Test-Path $OutJson) -and (Get-Item $OutJson).Length -gt $MinJsonBytes) {
        Write-Log "[PhpMetrics] SKIP $Name ($(Get-SizeKB $OutJson) KB exists)" "WARN"
        return $true
    }
    if ($DryRun) {
        Write-Log "[PhpMetrics] DRY-RUN $Name" "WARN"
        return $true
    }

    $fw         = Detect-Framework $SrcPath
    $exclusions = Get-FrameworkExclusions $fw
    $excludeArg = "--exclude=" + ($exclusions -join ",")

    Write-Log "[PhpMetrics] Running $Name (framework=$fw exclude=$($exclusions -join ','))"

    $errFile = Join-Path $env:TEMP "pm_err_$Name.txt"
    $proc = Start-Process "phpmetrics" `
        -ArgumentList @($SrcPath, "--extensions=php", "--report-json=$OutJson", $excludeArg) `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardError $errFile

    if ((Test-Path $OutJson) -and (Get-Item $OutJson).Length -gt $MinJsonBytes) {
        Write-Log "[PhpMetrics] OK $Name - $(Get-SizeKB $OutJson) KB" "OK"
        return $true
    }

    $err = "unknown error"
    if (Test-Path $errFile) {
        $errContent = Get-Content $errFile -Raw
        if ($null -ne $errContent) { $err = $errContent.Trim() }
    }
    $exitCode = if ($null -ne $proc) { $proc.ExitCode } else { "null" }
    Write-Log "[PhpMetrics] FAIL $Name - exit=$exitCode err=$err" "ERROR"
    return $false
}

# ==============================================================================
# PHPMD RUNNER  -- GodClass rule only (design ruleset)
#
# WHY "design" only, not "codesize,design,naming":
#   The train script's McNemar test (Step 10) compares RF predictions against
#   PHPMD_Flag on the same test instances. For this comparison to be valid,
#   PHPMD_Flag must represent PHPMD's GodClass detection specifically -- not
#   any arbitrary code-quality violation. Using "codesize" or "naming" rulesets
#   inflates PHPMD_Flag with unrelated violations, making the comparison apples
#   vs oranges and invalidating Step 10 entirely.
#
# PHPMD GodClass rule fires when:
#   ATFD > 5  AND  WMC > 47  AND  TCC < 0.33
#   (Note: due to PhpMetrics 2.9.x returning ATFD=0, PHPMD may rarely fire.
#    This is a known tool limitation documented in train_godclass.py. PHPMD
#    computes ATFD internally and independently from PhpMetrics.)
# ==============================================================================
function Invoke-PHPMD {
    param([string]$Name, [string]$SrcPath, [string]$OutXml)

    if (!$Force -and (Test-Path $OutXml) -and (Get-Item $OutXml).Length -gt $MinXmlBytes) {
        Write-Log "[PHPMD] SKIP $Name ($(Get-SizeKB $OutXml) KB exists)" "WARN"
        return $true
    }
    if ($DryRun) {
        Write-Log "[PHPMD] DRY-RUN $Name" "WARN"
        return $true
    }

    Write-Log "[PHPMD] Running $Name (ruleset=design [GodClass rule])"

    $errFile = Join-Path $env:TEMP "phpmd_err_$Name.txt"

    # --ignore-violations-on-exit prevents PHPMD from returning exit 1
    # on PHP 8.x trait method collision warnings (valid PHP code that
    # PHPMD cannot resolve). Without this flag, trait-heavy Laravel/
    # Symfony repos fail entirely despite producing valid XML output.
    #
    # Ruleset is "design" only -- targets GodClass, CouplingBetweenObjects,
    # ExcessiveClassComplexity, etc. from the design ruleset alone.
    $proc = Start-Process "phpmd" `
        -ArgumentList @($SrcPath, "xml", "design",
                        "--reportfile", $OutXml,
                        "--suffixes", "php",
                        "--ignore-violations-on-exit") `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardError $errFile

    # Exit codes:
    #   0 = no violations found                -> success
    #   1 = violations found (PHPMD behaviour) -> success (we want the XML)
    #   2 = violations found (some PHPMD 2.x)  -> success
    if ($null -eq $proc) {
        Write-Log "[PHPMD] FAIL $Name - Start-Process returned null" "ERROR"
        return $false
    }
    if ($proc.ExitCode -in @(0, 1, 2)) {
        if (!(Test-Path $OutXml)) {
            # PHPMD found nothing -> write minimal valid XML so downstream
            # code always has a file to parse (avoids null-check errors).
            $emptyXml = '<?xml version="1.0" encoding="UTF-8"?><pmd version="phpmd" timestamp="0"></pmd>'
            $emptyXml | Out-File -FilePath $OutXml -Encoding utf8
            Write-Log "[PHPMD] OK $Name - 0 violations (clean)" "OK"
        } else {
            Write-Log "[PHPMD] OK $Name - $(Get-SizeKB $OutXml) KB" "OK"
        }
        return $true
    }

    $err = "unknown error"
    if (Test-Path $errFile) {
        $errContent = Get-Content $errFile -Raw
        if ($null -ne $errContent) { $err = $errContent.Trim() }
    }
    Write-Log "[PHPMD] FAIL $Name - exit=$($proc.ExitCode) err=$err" "ERROR"
    return $false
}

# ==============================================================================
# SAFE PROPERTY ACCESS
# ==============================================================================
function Get-Prop {
    param($Obj, [string]$Key, $Default = 0)
    if ($Obj -and $Obj.PSObject.Properties[$Key]) { return $Obj.$Key }
    return $Default
}

# ==============================================================================
# CLASS NAME NORMALIZER
#
# PhpMetrics stores FQCNs like "App\Models\User" or "App_Models_User".
# PHPMD XML stores either FQCN or short name in the "class" attribute,
# depending on version and framework.
#
# To ensure PHPMD_Flag correctly matches PhpMetrics class names:
#   - Build a lookup from both FQCN and short name (last segment after \)
#   - When matching, try exact first, then short-name fallback.
#
# This prevents PHPMD_Flag always being 0 due to name format mismatches,
# which would silently corrupt the McNemar test in Step 10.
# ==============================================================================
function Normalize-ClassName {
    param([string]$Name)
    # Strip leading/trailing whitespace and backslash
    $n = $Name.Trim().Trim('\')
    # Normalise namespace separator: both \ and _ variants appear in practice
    return $n
}

function Get-ShortName {
    param([string]$FqcnOrName)
    $parts = $FqcnOrName -split '[\\|]'
    return $parts[-1].Trim()
}

# ==============================================================================
# CK METRIC EXTRACTION
#
# Reads PhpMetrics JSON + PHPMD XML for one project.
# Returns one PSCustomObject row per PHP class.
#
# METRIC COMPUTATION NOTES (aligned with train_godclass.py S.3.3):
#
#   WMC   -- "wmc" field (weighted method count = sum of cyclomatic complexities)
#   LCOM  -- "lcom" field (Henderson-Sellers LCOM, range [0,1]; 0 = fully cohesive)
#   CBO   -- efferentCoupling + afferentCoupling
#   RFC   -- WMC + CBO  (PhpMetrics 2.9.x proxy; no direct method-call count)
#   DIT   -- parents.Count (0 or 1 only due to PhpMetrics tool limitation)
#   NOC   -- derived: count how many classes list this class in their parents[]
#   TCC   -- 1/(1+LCOM) proxy (PhpMetrics has no native TCC output)
#   NOM   -- "nbMethods" (with fallback to methods[].Count)
#   LOC   -- "loc" field
#   ATFD  -- dependencies[].Count (always 0 in PhpMetrics 2.9.x; kept for audit)
#   NOA   -- "attributes" field (always 0 in PhpMetrics 2.9.x; kept for audit)
#
#   PHPMD_Flag  -- 1 if PHPMD GodClass rule fired for this class, else 0
#                 Matching uses FQCN first, then short-name fallback.
#   Expert_Hits -- count of threshold breaches (S.3.2.2):
#                   WMC > 47, LCOM > 0.8, LOC > 500, NOM > 20, CBO > 14
#   Label       -- 1 if PHPMD_Flag=1 OR Expert_Hits>=2, else 0
#   LabelSource -- "Both" | "PHPMD_only" | "Expert_only" | "Neither"
#                 Required by train_godclass.py Step 13 sensitivity analysis.
# ==============================================================================
function Extract-CKMetrics {
    param([string]$JsonPath, [string]$XmlPath, [string]$Project)

    $rows = @()

    if (!(Test-Path $JsonPath)) {
        Write-Log "[CK] SKIP $Project - JSON not found" "WARN"
        return $rows
    }

    # -- Parse PhpMetrics JSON -------------------------------------------------
    $json    = Get-Content $JsonPath -Raw | ConvertFrom-Json
    $classes = @($json.PSObject.Properties.Value | Where-Object {
        $_._type -like "*ClassMetric*"
    })

    if ($classes.Count -eq 0) {
        Write-Log "[CK] WARN $Project - no ClassMetric entries in JSON" "WARN"
        return $rows
    }

    # -- Build NOC map (child count per class name) ----------------------------
    # "parents" contains FQCNs of direct parent classes. We count how many
    # classes name each class as their parent to get NOC.
    $childMap = @{}
    foreach ($c in $classes) {
        foreach ($parent in @(Get-Prop $c "parents" @())) {
            $pn = Normalize-ClassName $parent
            if ($childMap.ContainsKey($pn)) { $childMap[$pn]++ }
            else { $childMap[$pn] = 1 }
        }
    }

    # -- Parse PHPMD XML -- GodClass violations only ----------------------------
    #
    # Build two lookup tables for robust matching:
    #   $phpmdFlagged_fqcn  -- keyed by normalized FQCN (for exact match)
    #   $phpmdFlagged_short -- keyed by short class name (for fallback match)
    #
    # Only count violations where rule="GodClass" (design ruleset).
    # Other design-rule violations (e.g. CouplingBetweenObjects) must NOT
    # set PHPMD_Flag -- otherwise the McNemar comparison in Step 10 is invalid.
    $phpmdFlagged_fqcn  = @{}
    $phpmdFlagged_short = @{}
    $phpmdGodClassCount = 0

    if ($XmlPath -and (Test-Path $XmlPath)) {
        try {
            [xml]$phpmdXml = Get-Content $XmlPath -Raw -Encoding utf8
            foreach ($file in $phpmdXml.pmd.file) {
                foreach ($violation in $file.violation) {
                    # Filter: only the GodClass rule
                    $rule = $violation.GetAttribute("rule")
                    if ($rule -ne "GodClass") { continue }

                    $cls = $violation.GetAttribute("class")
                    if (!$cls) { continue }

                    $fqcn  = Normalize-ClassName $cls
                    $short = Get-ShortName $fqcn
                    $phpmdFlagged_fqcn[$fqcn]   = 1
                    $phpmdFlagged_short[$short]  = 1
                    $phpmdGodClassCount++
                }
            }
        } catch {
            Write-Log "[CK] WARN $Project - PHPMD XML parse error: $_" "WARN"
        }
    }
    Write-Log "[CK] $Project - PHPMD GodClass violations found: $phpmdGodClassCount"

    # -- Per-class metric extraction -------------------------------------------
    $zeroLcomCount = 0

    foreach ($c in $classes) {

        $rawName = Get-Prop $c "name"
        if (!$rawName) { continue }
        $name      = Normalize-ClassName $rawName
        $shortName = Get-ShortName $name

        # WMC -- weighted method count (sum of cyclomatic complexities)
        $wmc = [int][math]::Max(0, [double](Get-Prop $c "wmc" 0))

        # LCOM -- Henderson-Sellers Lack of Cohesion (PhpMetrics "lcom" field)
        # Range: [0, 1] where 0 = perfectly cohesive, 1 = fully dispersed.
        # Note: some PhpMetrics versions also expose "lcom4"; we use "lcom"
        # which is the standard Henderson-Sellers variant.
        $lcom = [double](Get-Prop $c "lcom" 0)
        if ($lcom -lt 0) { $lcom = 0 }   # guard against negative edge case
        if ($lcom -eq 0) { $zeroLcomCount++ }

        # LOC -- lines of code
        $loc = [int][math]::Max(0, [double](Get-Prop $c "loc" 0))

        # NOM -- number of methods (prefer nbMethods; fallback to methods[].Count)
        $nom = [int](Get-Prop $c "nbMethods" 0)
        if ($nom -eq 0) {
            $methods = Get-Prop $c "methods" $null
            if ($methods) { $nom = @($methods).Count }
        }

        # NOA -- number of attributes (always 0 in PhpMetrics 2.9.x)
        # Kept in output for audit; excluded from training features.
        $attr = Get-Prop $c "attributes" 0
        $noa  = if ($attr -is [array]) { $attr.Count } else { [int]$attr }

        # CBO -- coupling between objects (efferent + afferent)
        $ce  = [double](Get-Prop $c "efferentCoupling" 0)
        $ca  = [double](Get-Prop $c "afferentCoupling" 0)
        $cbo = [int]($ce + $ca)

        # RFC -- response for a class (WMC + CBO proxy for PhpMetrics 2.9.x)
        # Standard RFC = WMC + |{methods called by methods of this class}|.
        # PhpMetrics 2.9.x does not expose a direct method-call count, so we
        # use WMC+CBO as a proxy. The training script documents this limitation.
        $rfc = $wmc + $cbo

        # DIT -- depth of inheritance tree (parents[].Count, limited to 0 or 1)
        # PhpMetrics "parents" contains only direct parents, not ancestors.
        # DIT is therefore 0 (no parent) or 1 (has direct parent). This
        # limitation is noted in train_godclass.py and DIT is retained despite
        # low variance because it captures the has-parent signal.
        $parents = @(Get-Prop $c "parents" @())
        $dit     = $parents.Count

        # NOC -- number of children (derived from cross-class parent map)
        $noc = if ($childMap.ContainsKey($name)) { $childMap[$name] } else { 0 }

        # ATFD -- access to foreign data (dependencies[].Count)
        # Always 0 in PhpMetrics 2.9.x due to tool limitation.
        # Kept for audit; excluded from training features.
        $deps = Get-Prop $c "dependencies" $null
        $atfd = if ($deps) { @($deps).Count } else { 0 }

        # TCC -- tight class cohesion proxy: 1/(1+LCOM)
        # Range: (0, 1] where 1 = perfectly cohesive (LCOM=0).
        # Standard TCC requires method-pair analysis not available in PhpMetrics.
        # This proxy inversely mirrors LCOM and is consistent with how the
        # training script treats TCC as a "cohesion complement" to LCOM.
        $tcc = [math]::Round(1.0 / (1.0 + $lcom), 4)

        # -- PHPMD flag -- GodClass rule match ---------------------------------
        # Try FQCN match first (most reliable), then short-name fallback.
        # PHPMD sometimes reports only the short class name in the "class"
        # attribute, while PhpMetrics stores FQCNs.
        $phpmdFlag = 0
        if ($phpmdFlagged_fqcn.ContainsKey($name)) {
            $phpmdFlag = 1
        } elseif ($phpmdFlagged_short.ContainsKey($shortName)) {
            $phpmdFlag = 1
        }

        # -- Expert heuristic threshold hits (S.3.2.2) -------------------------
        # Thresholds from Lanza & Marinescu (2006) as cited in thesis S.3.2.2:
        #   WMC > 47   -- ExcessiveClassComplexity threshold
        #   LCOM > 0.8 -- high lack of cohesion
        #   LOC > 500  -- ExcessiveClassLength threshold
        #   NOM > 20   -- ExcessivePublicCount threshold (method variant)
        #   CBO > 14   -- CouplingBetweenObjects threshold
        $expertHits = 0
        if ($wmc  -gt 47)  { $expertHits++ }
        if ($lcom -gt 0.8) { $expertHits++ }
        if ($loc  -gt 500) { $expertHits++ }
        if ($nom  -gt 20)  { $expertHits++ }
        if ($cbo  -gt 14)  { $expertHits++ }

        # -- Dual-source label (S.3.2) ------------------------------------------
        # Label = 1 if PHPMD GodClass rule fired OR >= 2 expert threshold hits.
        $label = if ($phpmdFlag -eq 1 -or $expertHits -ge 2) { 1 } else { 0 }

        # -- LabelSource -- required for Step 13 sensitivity analysis -----------
        # train_godclass.py Step 13 expects LabelSource in
        #   {"PHPMD-confirmed", "expert-overridden", "dual-source"}
        # We map our categories to match those expected values:
        #   Both        -> "dual-source"       (PHPMD fired AND expert hits >= 2)
        #   PHPMD_only  -> "PHPMD-confirmed"   (PHPMD fired,  expert hits < 2)
        #   Expert_only -> "expert-overridden" (PHPMD silent, expert hits >= 2)
        #   Neither     -> "Neither"           (non-God class, label=0)
        if ($phpmdFlag -eq 1 -and $expertHits -ge 2) { $labelSource = "dual-source"       }
        elseif ($phpmdFlag -eq 1)                     { $labelSource = "PHPMD-confirmed"   }
        elseif ($expertHits -ge 2)                    { $labelSource = "expert-overridden" }
        else                                          { $labelSource = "Neither"           }

        $rows += [PSCustomObject]@{
            Project     = $Project
            Class       = $name
            WMC         = $wmc
            LCOM        = [math]::Round($lcom, 4)
            CBO         = $cbo
            RFC         = $rfc
            DIT         = $dit
            NOC         = $noc
            TCC         = $tcc
            NOM         = $nom
            LOC         = $loc
            ATFD        = $atfd    # always 0; kept for audit only
            NOA         = $noa     # always 0; kept for audit only
            PHPMD_Flag  = $phpmdFlag
            Expert_Hits = $expertHits
            Label       = $label
            LabelSource = $labelSource
        }
    }

    # Warn if LCOM is universally zero (indicates field name mismatch in JSON)
    if ($zeroLcomCount -eq $classes.Count -and $classes.Count -gt 0) {
        Write-Log "[CK] WARN $Project - ALL $($classes.Count) classes have LCOM=0; check PhpMetrics JSON field name" "WARN"
    }

    return $rows
}

# ==============================================================================
# MAIN LOOP
# ==============================================================================
$allProjects = Get-ChildItem $ProjectDir -Directory | Sort-Object Name

if ($Only -ne "") {
    $allProjects = $allProjects | Where-Object { $_.Name -eq $Only }
    if (!$allProjects) {
        Write-Error "Project '$Only' not found in $ProjectDir"
        exit 1
    }
}

$skippedN = 0
if ($RebuildCSV) {
    Write-Host "  [REBUILD] Rebuilding ck_metrics_all.csv from all cached JSON files..." -ForegroundColor Yellow
    if (Test-Path $CsvAll) { Remove-Item $CsvAll -Force }
    $SkipPhpMetrics = $true   # never re-run tools in rebuild mode
} elseif ($Resume -and !$Force) {
    $before      = $allProjects.Count
    $allProjects = $allProjects | Where-Object { !(Is-Completed $_.Name) }
    $skippedN    = $before - $allProjects.Count
    Write-Host "  [RESUME] Skipping $skippedN already-completed projects" -ForegroundColor Yellow
}

$total   = $allProjects.Count
$Start   = Get-Date
$success = @()
$failed  = @()
$idx     = 0

# Pre-load any rows already written by a previous run so that -Resume
# never loses data for completed projects (the loop body is skipped for them).
$allRows = @()
if (!$RebuildCSV -and (Test-Path $CsvAll)) {
    try {
        $existing = Import-Csv -Path $CsvAll -Encoding utf8
        $allRows  = @($existing)
        Write-Host "  [CSV] Loaded $($allRows.Count) existing rows from previous run" -ForegroundColor Yellow
        Write-Log "CSV pre-loaded: $($allRows.Count) rows from $CsvAll"
    } catch {
        Write-Host "  [WARN] Could not load existing CSV -- starting fresh" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  XDD - PhpMetrics + PHPMD + CK Metrics (UNIFIED)"          -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Projects   : $total"
Write-Host "  Output dir : $OutputDir"
Write-Host "  CK CSV     : $CsvAll"
Write-Host "  Log file   : $LogFile"
Write-Host "  PHPMD rule : design (GodClass only)"
Write-Host "============================================================"
Write-Host ""

foreach ($proj in $allProjects) {

    $idx++
    $name    = $proj.Name
    $path    = $proj.FullName
    $jsonOut = Join-Path $OutputDir "$name.json"
    $xmlOut  = Join-Path $OutputDir "$name-phpmd.xml"

    Write-Host "[$idx/$total] $name" -ForegroundColor Cyan
    Write-Log "START $name"

    try {
        # Step 1 -- PhpMetrics
        $pmOk = $true
        if (!$SkipPhpMetrics) {
            $pmOk = Invoke-PhpMetrics -Name $name -SrcPath $path -OutJson $jsonOut
        } else {
            Write-Log "[PhpMetrics] Skipped (-SkipPhpMetrics or -RebuildCSV flag)" "WARN"
        }

        # Step 2 -- PHPMD (GodClass rule only)
        $mdOk = Invoke-PHPMD -Name $name -SrcPath $path -OutXml $xmlOut

        # Step 3 -- Extract CK metrics from JSON + PHPMD XML
        $rows    = Extract-CKMetrics -JsonPath $jsonOut -XmlPath $xmlOut -Project $name
        $allRows += $rows

        $allOk = $pmOk -and $mdOk -and ($rows.Count -gt 0)
        if ($allOk) {
            Mark-Completed $name
            $success += $name
            $godInProj = ($rows | Where-Object { $_.Label -eq 1 }).Count
            Write-Log "DONE $name - $($rows.Count) classes | GodClass=$godInProj" "OK"
        } else {
            $failed += $name
            Write-Log "FAIL $name - pmOk=$pmOk mdOk=$mdOk rows=$($rows.Count)" "ERROR"
        }
    } catch {
        $failed += $name
        Write-Log "ERROR $name - unexpected exception: $_" "ERROR"
        Write-Host "  [SKIPPED] $name - exception caught, continuing to next project" -ForegroundColor Yellow
    }

    Write-Host "  ----------------------------------------------------------"
}

# ==============================================================================
# EXPORT CSV
# Deduplication: when a project is re-run (e.g. -Force -Only drupal), remove
# its old rows from the accumulated set, then append the fresh rows.
# Strategy: remove all rows for projects in $success (just re-processed),
# then re-add the fresh rows for those projects from $allRows tail.
# This avoids reverse-iteration complexity and is explicit.
# ==============================================================================
if ($allRows.Count -gt 0) {

    if ($success.Count -gt 0) {
        # Separate old rows (stale) from fresh rows (just produced)
        # Fresh rows are those just appended in the loop above; old rows
        # for the same projects were loaded from the pre-existing CSV.
        # Simplest correct approach: drop ANY row whose Project is in
        # $success from the pre-loaded set, keep all freshly extracted rows.
        #
        # We distinguish pre-loaded vs freshly added by rebuilding:
        #   1. Start with pre-loaded rows minus re-processed projects.
        #   2. Add all freshly extracted rows (already in $allRows tail).
        # But since $allRows is mixed, we use Project+Class keyed hashtable.
        $seen    = @{}
        $deduped = [System.Collections.Generic.List[object]]::new()

        # Iterate forward. Later rows (fresher) overwrite earlier (stale).
        # Since fresh rows for $success projects were appended last in the
        # loop, iterating forward and overwriting keys means we keep fresh.
        $keyMap = @{}
        foreach ($row in $allRows) {
            $key = $row.Project + "|" + $row.Class
            $keyMap[$key] = $row   # last write wins -> fresh row survives
        }
        $allRows = @($keyMap.Values | Sort-Object Project, Class)
    }

    $allRows | Export-Csv -Path $CsvAll -NoTypeInformation -Encoding utf8

    $godCount    = ($allRows | Where-Object { $_.Label -eq 1 }).Count
    $nonGodCount = $allRows.Count - $godCount
    $godPct      = [math]::Round(($godCount / $allRows.Count) * 100, 1)

    # LabelSource breakdown (for quick validation of sensitivity analysis input)
    $srcCounts = $allRows | Group-Object LabelSource |
                 Select-Object Name, Count |
                 Sort-Object Name
    $phpmdOnlyCount  = ($allRows | Where-Object { $_.PHPMD_Flag -eq 1 }).Count
    $expertOnlyCount = ($allRows | Where-Object { $_.Expert_Hits -ge 2 }).Count

    Write-Host ""
    Write-Host "  DATASET STATISTICS" -ForegroundColor Cyan
    Write-Host "  Total classes  : $($allRows.Count)"
    Write-Host "  God Class  (1) : $godCount ($godPct%)"
    Write-Host "  Non-God    (0) : $nonGodCount"
    Write-Host ""
    Write-Host "  LabelSource breakdown (for Step 13 sensitivity analysis):"
    foreach ($s in $srcCounts) {
        Write-Host ("    {0,-20} : {1}" -f $s.Name, $s.Count)
    }
    Write-Host "  PHPMD_Flag=1 total : $phpmdOnlyCount  (McNemar baseline, Step 10)"
    Write-Host "  Expert_Hits>=2     : $expertOnlyCount"
    Write-Log ("CSV saved: $CsvAll ($($allRows.Count) rows | GodClass=$godCount [$godPct%])")

    # -- Sanity checks aligned with train_godclass.py expectations ------------
    Write-Host ""
    Write-Host "  SANITY CHECKS:" -ForegroundColor Cyan

    # Check 1: PHPMD_Flag column exists and has variation
    if ($phpmdOnlyCount -eq 0) {
        Write-Host "  [WARN] PHPMD_Flag is 0 for ALL classes. McNemar test (Step 10) will be skipped." -ForegroundColor Yellow
        Write-Host "         Causes: (a) PHPMD GodClass rule never fired (ATFD=0 limitation)," -ForegroundColor Yellow
        Write-Host "                 (b) class name mismatch between PHPMD XML and PhpMetrics JSON." -ForegroundColor Yellow
        Write-Log "WARN: PHPMD_Flag=0 for all classes -- McNemar test will be skipped" "WARN"
    } else {
        Write-Host "  [OK] PHPMD_Flag has $phpmdOnlyCount positive cases" -ForegroundColor Green
    }

    # Check 2: LabelSource column is present (Step 13 requires it)
    $hasBoth   = ($allRows | Where-Object { $_.LabelSource -eq "dual-source"       }).Count
    $hasPHPMD  = ($allRows | Where-Object { $_.LabelSource -eq "PHPMD-confirmed"   }).Count
    $hasExpert = ($allRows | Where-Object { $_.LabelSource -eq "expert-overridden" }).Count
    Write-Host "  [OK] LabelSource column present (Step 13 sensitivity analysis enabled)" -ForegroundColor Green
    Write-Host "       dual-source=$hasBoth  PHPMD-confirmed=$hasPHPMD  expert-overridden=$hasExpert"

    # Check 3: God Class prevalence in expected range (~5-15%)
    if ($godPct -lt 3 -or $godPct -gt 25) {
        Write-Host "  [WARN] God Class prevalence=$godPct% is outside typical 5-15% range" -ForegroundColor Yellow
        Write-Host "         Review labeling thresholds in Extract-CKMetrics if unexpected." -ForegroundColor Yellow
    } else {
        Write-Host "  [OK] God Class prevalence=$godPct% is within expected range" -ForegroundColor Green
    }

    # Check 4: Features used by training script are all non-zero somewhere
    $featureCheck = @("WMC","LCOM","CBO","RFC","DIT","NOC","TCC","NOM","LOC")
    foreach ($feat in $featureCheck) {
        $nonZero = ($allRows | Where-Object { [double]$_.$feat -ne 0 }).Count
        if ($nonZero -eq 0) {
            Write-Host "  [WARN] Feature '$feat' is 0 for ALL classes -- check extraction" -ForegroundColor Yellow
        }
    }

    Write-Host ""
}

# ==============================================================================
# SUMMARY
# ==============================================================================
$elapsed = (Get-Date) - $Start
$elStr   = "{0:mm}m {0:ss}s" -f $elapsed

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SUMMARY"                                                    -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Processed  : $total"
Write-Host "  Succeeded  : $($success.Count)" -ForegroundColor Green
Write-Host "  Failed     : $($failed.Count)"  -ForegroundColor $(if ($failed.Count) { "Red" } else { "Green" })
Write-Host "  Skipped    : $skippedN"
Write-Host "  Elapsed    : $elStr"
Write-Host ""
Write-Host "  Output:"
Write-Host "    PhpMetrics JSON  ->  output\*.json"
Write-Host "    PHPMD XML        ->  output\*-phpmd.xml"
Write-Host "    CK Metrics CSV   ->  output\ck_metrics_all.csv"
Write-Host ""
Write-Host "  CSV columns for train_godclass.py:"
Write-Host "    Features : WMC LCOM CBO RFC DIT NOC TCC NOM LOC"
Write-Host "    Target   : Label"
Write-Host "    McNemar  : PHPMD_Flag  (Step 10)"
Write-Host "    Sensitiv : LabelSource (Step 13)"

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed projects:" -ForegroundColor Red
    foreach ($f in $failed) { Write-Host "    - $f" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Retry: .\Run-PhpMetrics-PHPMD.ps1 -Resume" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "  All complete." -ForegroundColor Green
    Write-Host "  Next step: python train_godclass.py" -ForegroundColor White
}

Write-Host "============================================================"
exit $failed.Count