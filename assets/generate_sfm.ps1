if (Test-Path -Path ./build -PathType Container) {
	#rmdir build -r
	exit
}

mkdir ./build
mkdir ./build/images
Get-ChildItem -Path ./ -Filter "cam_*.png" | Where-Object { $_.Name -match '^cam_0[1-9]\.png$' -or $_.Name -match '^cam_1[0-2]\.png$' } | ForEach-Object { Copy-Item -Path $_.FullName -Destination ./build/images }
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_SfMInit_ImageListing.exe -i ./build/images -o build -f 10200 -g 0
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_ComputeFeatures.exe -i build/sfm_data.json -o build
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_PairGenerator.exe -i build/sfm_data.json -o build/pairs.txt -m EXHAUSTIVE
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_ComputeMatches.exe -i build/sfm_data.json -o build/matches.putative.bin -p build/pairs.txt
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_GeometricFilter.exe -i build/sfm_data.json -m build/matches.putative.bin -o build/matches.geom.bin
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_SfM.exe -i build/sfm_data.json -o build -M build/matches.geom.bin --sfm_engine INCREMENTAL
C:\Users\Admin\Desktop\openmvg_openmvs\openMVG\build\Windows-AMD64-Release\Release\openMVG_main_ConvertSfM_DataFormat.exe -i build/sfm_data.bin -o build/sfm_data_conf.json -V -I -E
python C:\Users\Admin\Desktop\master_thesis\volume_prediction_fip\assets\build_conf_openmvg.py build/sfm_data_conf.json poses.json

Get-ChildItem -Path ./build -File | Where-Object { $_.Name -notin "sfm_data.bin", "SfMReconstruction_Report.html" } | Remove-Item -Force
rmdir build/images -r