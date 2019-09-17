# execute in videos folder
ls -1 > framerates0.txt

echo "fetching framerate..."
for i in `cat framerates0.txt`; do ffmpeg -i $i 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"; done > framerates1.txt

paste -d ' ' framerates0.txt framerates1.txt > ../framerates.txt
rm framerates0.txt framerates1.txt