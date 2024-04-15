# 获取所有硬盘信息
$diskDrives = Get-WmiObject Win32_DiskDrive

# 显示所有硬盘信息
$diskDrives | Format-Table DeviceID, Model, Size

# 选择要操作的硬盘
$disk = $diskDrives | Where-Object { $_.DeviceID -eq "YOUR_DISK_ID" }

# 例如，执行分区操作
# $partition = $disk.CreatePartition($size) # $size 是要创建的分区大小

# 获取 C 盘信息
$disk = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'"

# 显示 C 盘的现存空间
$freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
$totalSizeGB = [math]::Round($disk.Size / 1GB, 2)

Write-Host "C 盘剩余空间：$freeSpaceGB GB"
Write-Host "C 盘总空间：$totalSizeGB GB"

# 获取所有进程信息
$processes = Get-WmiObject Win32_Process

# 获取所有现存分区的盘符
$partitions = Get-WmiObject Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 }

# 显示所有现存分区的盘符
$partitions | ForEach-Object { $_.DeviceID }

# 获取要修改的分区信息（例如，将 D 盘改为 E 盘）
$partition = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='D:'"

# 设置新的盘符
#$partition.DeviceID = "E:"

# 更新分区的盘符
#$partition.Put()

# 获取所有无盘符的分区信息
$partitions = Get-WmiObject Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 -and $_.VolumeName -eq $null }

# 如果没有找到无盘符的分区，退出脚本
if ($partitions.Count -eq 0) {
    Write-Host "没有找到无盘符的分区"
    exit
}

# 找到可用容量最大的分区
$maxFreeSpace = 0
$maxFreeSpacePartition = $null

foreach ($partition in $partitions) {
    if ($partition.FreeSpace -gt $maxFreeSpace) {
        $maxFreeSpace = $partition.FreeSpace
        $maxFreeSpacePartition = $partition
    }
}

# 如果找到可用容量最大的分区，为其分配盘符"N"
if ($maxFreeSpacePartition -ne $null) {
    $maxFreeSpacePartition.DeviceID = "N:"
    $maxFreeSpacePartition.Put()
    Write-Host "已将盘符 N 分配给分区：" $maxFreeSpacePartition.DeviceID
} else {
    Write-Host "未找到符合条件的分区"
}


# 获取所有分区信息
$partitions = Get-WmiObject Win32_DiskPartition

# 显示所有分区的详细信息，包括盘符
$partitions | ForEach-Object {
    $partition = $_
    $disks = Get-WmiObject -Query "ASSOCIATORS OF {Win32_DiskPartition.DeviceID='$($partition.DeviceID)'} WHERE AssocClass = Win32_LogicalDiskToPartition"
    $disks | ForEach-Object {
        $disk = $_
        [PSCustomObject]@{
            DeviceID = $disk.DeviceID
            PartitionDeviceID = $partition.DeviceID
            Name = $partition.Name
            Size = $partition.Size
            Type = $partition.Type
        }
    }
} | Format-Table DeviceID, PartitionDeviceID, Name, Size, Type

Set-ExecutionPolicy RemoteSigned -Scope Process
Set-ExecutionPolicy RemoteSigned -Scope Process
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
Set-ExecutionPolicy Unrestricted -Scope CurrentUser
