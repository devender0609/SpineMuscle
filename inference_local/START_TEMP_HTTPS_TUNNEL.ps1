$ErrorActionPreference="Stop"
Write-Host "SpineMuscle AI - temporary HTTPS tunnel" -ForegroundColor Cyan
$cf=(Get-Command cloudflared -ErrorAction SilentlyContinue)
if(!$cf){throw "cloudflared was not found in PATH."}
Write-Host "Keep this second window open. Copy the https://...trycloudflare.com URL." -ForegroundColor Yellow
cloudflared tunnel --url http://127.0.0.1:8080
