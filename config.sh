if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	export LD_LIBRARY_PATH="$(pwd)/node_modules/\@arition/torch-js/build/libtorch:$(pwd)/node_modules/\@arition/torch-js/build/vision"
elif [[ "$OSTYPE" == "darwin"* ]]; then
	export DYLD_LIBRARY_PATH="$(pwd)/node_modules/\@arition/torch-js/build/libtorch:$(pwd)/node_modules/\@arition/torch-js/build/vision"
fi