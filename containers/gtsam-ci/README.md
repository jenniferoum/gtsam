This folder contains Dockerfiles for building containers used in the CI/CD steps of gtsam.

Naming convention: `Dockerfile.{os}.{compiler}`

If any changes are made to this folder/sub-folders, remember to build and push the new docker images using:
```
    ./build_and_push.sh
```