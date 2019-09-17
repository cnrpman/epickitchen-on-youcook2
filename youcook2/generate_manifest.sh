CURRENT=`pwd`
BASENAME=`basename "$CURRENT"`

# Execute under videos folder
ls -1 > ../manifest1.txt
mv ../manifest1.txt ./

echo "fetching tot frame..."
for i in `cat manifest1.txt`; do ls -1 -- $i | wc -l; done > manifest2.txt

echo "generating psuedo label..."
for i in `cat manifest1.txt`; do echo 0; done > manifest3.txt

sed -i "s/^/$BASENAME\//" manifest1.txt
paste -d' ' manifest1.txt manifest2.txt manifest3.txt > manifest_$BASENAME.txt
rm manifest1.txt manifest2.txt manifest3.txt

mv manifest_$BASENAME.txt ../