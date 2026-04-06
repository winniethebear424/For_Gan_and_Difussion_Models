@echo off
del assignment_submission.zip 2>nul
tar -a -c -f assignment_submission.zip ^
  configs models\*.py losses\*.py utils\*.py ^
  outputs\vae\*.pth outputs\gan\*.pth outputs\diffusion\*.pth ^
  *.py
echo Zip created: assignment_submission.zip
