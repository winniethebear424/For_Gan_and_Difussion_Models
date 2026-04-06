rm -f assignment_submission.zip
zip -r assignment_submission.zip  *.py configs models/*py losses/*py utils/*.py   outputs/vae/*.pth outputs/gan/*.pth outputs/diffusion/*.pth *.py
# configs models/*py losses/*py utils/*.py   outputs/vae/*.pth outputs/gan/*.pth outputs/diffusion/*.pth *.py
# trainer_vae.py trainer_gan.py trainer_diffusion.py
