
param(
    [Parameter(Mandatory=$true)]
    [string]$DrivePath = "C:\",
    [string]$OutputCsv = "DiskUsageReport.csv"
)

# 初始化数据结构
$resultTable = New-Object System.Collections.ArrayList
$dirSizes = @{}

# 获取所有项目（包含系统/隐藏文件）
$allItems = Get-ChildItem -Path $DrivePath -Recurse -Force -ErrorAction SilentlyContinue

# 第一阶段：构建表结构
foreach ($item in $allItems) {
    if ($item.PSIsContainer) {
        $dirSizes[$item.FullName] = 0
    } else {
        $fileSize = (Get-Item $item.FullName -Force).Length
        $dirSizes[$item.Directory.FullName] += $fileSize

        $pathParts = $item.FullName.Replace($DrivePath, "").Split('\', [StringSplitOptions]::RemoveEmptyEntries)
        $row = [PSCustomObject]@{FullPath=$item.FullName}
        
        for ($i=0; $i -lt ($pathParts.Count-1); $i++) {
            $row | Add-Member "Dir_$($i+1)_Name" $pathParts[$i]
            $row | Add-Member "Dir_$($i+1)_Size" $null
        }
        
        $row | Add-Member "FileName" $item.Name
        $row | Add-Member "FileSize" $fileSize
        [void]$resultTable.Add($row)
    }
}

# 第二阶段：填充目录大小
foreach ($row in $resultTable) {
    $parentDir = [System.IO.Path]::GetDirectoryName($row.FullPath)
    while ($parentDir -and $parentDir -ne $DrivePath) {
        $dirName = [System.IO.Path]::GetFileName($parentDir)
        $sizeProp = ($row.PSObject.Properties | Where-Object {
            $_.Name -match "Dir_\d+_Name" -and $_.Value -eq $dirName
        }).Name -replace 'Name','Size'
        
        if ($sizeProp) { $row.$sizeProp = $dirSizes[$parentDir] }
        $parentDir = [System.IO.Path]::GetDirectoryName($parentDir)
    }
}

# 输出结果（修正编码问题）
$resultTable | Select-Object * -ExcludeProperty FullPath | Export-Csv -Path $OutputCsv -NoTypeInformation
Write-Host "Report generated: $((Get-Item $OutputCsv).FullName)"
