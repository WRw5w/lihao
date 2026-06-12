@echo off
setlocal

set "BAIDUPCS_GO_CONFIG_DIR=%~dp0config"
if not exist "%BAIDUPCS_GO_CONFIG_DIR%" mkdir "%BAIDUPCS_GO_CONFIG_DIR%"

"%~dp0BaiduPCS-Go-v3.6.2-windows-x64\BaiduPCS-Go.exe" %*
