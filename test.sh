#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="4g"

docker volume create nlst_monai-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v nlst_monai-output-$VOLUME_SUFFIX:/output/ \
        nlst_monai

#CAVE: We only check for correct size of 224x192x224x3
docker run --rm \
        -v nlst_monai-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        insighttoolkit/simpleitk-notebooks:latest python -c "import SimpleITK, numpy; pred = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage('/output/images/displacement-field/thisIsAnArbitraryFilename.mha')); ref = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage('/input/reference_disp_0101_0101.mha')); assert numpy.allclose(pred, ref, atol=1e-2)"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm nlst_monai-output-$VOLUME_SUFFIX