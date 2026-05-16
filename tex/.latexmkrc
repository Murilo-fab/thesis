$pdf_mode = 1;      # Force PDF generation
$out_dir = 'build'; # Output all files to 'build/' folder

# OPTIONAL: 
# If you want the final PDF to be copied back to the main directory
# (so you don't have to dig into build/ to find it)
$post_system = 'copy "build\\%R.pdf" "."'; 
# Note: On Mac/Linux, use 'cp' instead of 'copy' and forward slashes.