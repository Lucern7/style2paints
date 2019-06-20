BASE64_IMAGE=$(base64 -i dog.jpg | xargs echo "data:image/jpeg;base64," | sed "s/ //" )

# Make a POST request to the /classify command, receiving
curl http://0.0.0.0:8000/paint \
   -X POST \
   -H "content-type: application/json" \
   -d "{ \"input_image\": \"${BASE64_IMAGE}\" }"