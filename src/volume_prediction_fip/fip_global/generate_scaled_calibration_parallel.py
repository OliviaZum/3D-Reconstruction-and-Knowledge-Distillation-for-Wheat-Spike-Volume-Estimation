import os
import re
import csv
import shutil
import subprocess
from datetime import datetime

# === Configuration ===
EXTRINSICS_DIR = "../assets/fip_calibration/extrinsics"
PLOTS_INPUT_DIR = "../../../../data-kp/FIP/Analysis/2024/WW036/debayered"
PLOTS_OUTPUT_DIR = "../../../../data-kp/FIP/Analysis/2023/WW034/volume_prediction/2024"
FIP_CALIBRATE_EXECUTABLE = "fip-calibrate"  # Make sure it's in PATH or use full path
OPENMVG_PATH = "/home/zumstego/volume_prediction_fip/tools/openMVG/build/Linux-x86_64-RELEASE"
POSES_FILE_NAME = "poses_scaled.json"

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_CSV = f"calibration_summary_{now_str}.csv"

print("paths")
print(PLOTS_INPUT_DIR)
print(PLOTS_OUTPUT_DIR)



def extract_plot_date(plot_name):
    match = re.search(r'_(\d{8})_', plot_name)
    return datetime.strptime(match.group(1), "%Y%m%d") if match else None

def extract_extrinsic_date(fname):
    match = re.match(r'(\d{4}_\d{2}_\d{2})', fname)
    return datetime.strptime(match.group(1), "%Y_%m_%d") if match else None

def find_latest_extrinsic_file(plot_date):
    candidates = []
    for fname in os.listdir(EXTRINSICS_DIR):
        if fname.endswith(".txt"):
            date = extract_extrinsic_date(fname)
            if date and date <= plot_date:
                candidates.append((date, fname))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]

def run_calibration(plot_name):
    input_folder = os.path.join(PLOTS_INPUT_DIR, plot_name)
    output_folder = os.path.join(PLOTS_OUTPUT_DIR, plot_name)
    poses_file = os.path.join(output_folder, POSES_FILE_NAME)

    extrinsic_file = ""  


    if os.path.exists(poses_file):
        return "skipped", "", extrinsic_file

    plot_date = extract_plot_date(plot_name)
    if not plot_date:
        return "error", "invalid_plot_date", extrinsic_file

    extrinsic_file = find_latest_extrinsic_file(plot_date)
    if not extrinsic_file:
        return "error", "no_extrinsics_found", extrinsic_file

    os.makedirs(output_folder, exist_ok=True)

    try:
        subprocess.run([
            FIP_CALIBRATE_EXECUTABLE,
            "-o", OPENMVG_PATH,
            "-e", EXTRINSICS_DIR,
            "-p", output_folder,
            "-l", input_folder
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)

        if os.path.exists(poses_file):
            return "ok", "", extrinsic_file
        else:
            return "incomplete", "missing_output", extrinsic_file
    except subprocess.TimeoutExpired:
        return "error", "timeout", extrinsic_file
    except Exception as e:
        return "error", str(e), extrinsic_file

def find_fallback_source(failed_plot, failed_date, existing_poses_by_date):
    parent = os.path.dirname(failed_plot)
    candidates = existing_poses_by_date.get(failed_date, {})

    if parent in candidates:
        return candidates[parent]["path"], candidates[parent]["source_plot"]

    return None, None


def main():
    

    plot_folders = []

    for date_folder in os.listdir(PLOTS_INPUT_DIR):
        date_match = re.match(r'^(\d{4})_(\d{2})_(\d{2})', date_folder)
        if not date_match:
            continue  # skip folders without valid date prefix
        
        year, month, day = map(int, date_match.groups())
        folder_date = datetime(year, month, day)


        if folder_date < datetime(2024, 5, 15):
            continue  # skip anything before May

        full_date_path = os.path.join(PLOTS_INPUT_DIR, date_folder)

        if not os.path.isdir(full_date_path):
            continue

        # Now look for valid plot folders inside
        for plot_name in os.listdir(full_date_path):
            if re.match(r'^FPWW\d+_FIP2_\d{8}_\d{6}$', plot_name):
                rel_path = os.path.relpath(os.path.join(full_date_path, plot_name), PLOTS_INPUT_DIR)
                plot_folders.append(rel_path)


    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["plot", "date", "status", "reason", "extrinsics_file"])
        writer.writeheader()
            

    
    print("done step 1: finding all plots")
    summary = []
    existing_poses_by_date = {}

    # First pass: try to calibrate
    print("step 2: calibrating first round")
    for plot in plot_folders:
        status, reason, extrinsic_file = run_calibration(plot)
        date = extract_plot_date(plot).strftime("%Y%m%d") if extract_plot_date(plot) else "unknown"

        if status in ("ok", "skipped"):
            parent_folder = os.path.dirname(plot)
            if date not in existing_poses_by_date:
                existing_poses_by_date[date] = {}

            existing_poses_by_date[date][parent_folder] = {
            "path": os.path.join(PLOTS_OUTPUT_DIR, plot, POSES_FILE_NAME),
            "source_plot": plot  # Save the full relative plot path
             }

        row = {
            "plot": plot,
            "status": status,
            "reason": reason,
            "date": date,
            "extrinsics_file": extrinsic_file
        }

        # Write one row at a time
        with open(SUMMARY_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["plot", "date", "status", "reason", "extrinsics_file"])
            writer.writerow(row)

        print(f"{plot}: {status} ({reason})" if reason else f"{plot}: {status}")

    
    
    # Second pass: copy fallback poses_scaled.json where missing
    print("step 2: finding latest summary (fallback)")

    # Load the current summary file (which includes all processed plots)
    print(f"Loading fallback source info from current summary: {SUMMARY_CSV}")
    fallback_summary = []
    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fallback_summary = list(reader)


    print("step 2: copying files")
    for entry in fallback_summary:
        if entry["status"] in ("incomplete", "error"):
            target_folder = os.path.join(PLOTS_OUTPUT_DIR, entry["plot"])
            target_poses = os.path.join(target_folder, POSES_FILE_NAME)

            #  Always define fallback and fallback_parent
            fallback, fallback_parent = find_fallback_source(entry["plot"], entry["date"], existing_poses_by_date)

            if fallback:
                try:
                    if not os.path.exists(target_poses):
                        shutil.copy(fallback, target_poses)
                        entry["status"] = "copied_fallback"
                        entry["reason"] = f"copied_from: {fallback_parent}"
                        print(f"{entry['plot']}: copied poses_scaled.json from {fallback_parent}")
                    else:
                        entry["status"] = "used_existing_fallback"
                        entry["reason"] = f"poses already present (assumed fallback)"

                    entry["extrinsics_file"] = os.path.basename(fallback)


                    with open(SUMMARY_CSV, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["plot", "date", "status", "reason", "extrinsics_file"])
                        writer.writerow(entry)

                    #delete the .build folder, takes too much space
                    build_dir = os.path.join(target_folder, ".build")
                    if os.path.exists(build_dir):
                        try: 
                            shutil.rmtree(build_dir)
                        except Exception as e:
                            print(f"Could not remove {build_dir}: {e}")




                    #  Cleanup .build regardless
                    #build_dir = os.path.join(target_folder, ".build")
                    #if os.path.exists(build_dir):
                    #    for filename in os.listdir(build_dir):
                    #        file_path = os.path.join(build_dir, filename)
                    #        if (
                    #            filename.endswith(('.ply', '.desc', '.feat', '.svg', '.txt')) or
                    #            filename in {
                    #                "geometric_matches", "image_describer.json",
                    #                "matches.geom.bin", "matches.putative.bin",
                    #                "Reconstruction_Report.html", "sfm_data.json"
                    #            }
                    #        ):
                    #            try:
                    #                os.remove(file_path)
                    #                #print(f"Deleted: {file_path}")
                    #            except Exception as e:
                    #                print(f"Could not remove {file_path}: {e}")

                except Exception as e:
                    entry["reason"] = f"fallback_copy_failed: {str(e)}"




   

if __name__ == "__main__":
    main()

