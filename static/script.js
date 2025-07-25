const handleSubmit = () => {
    const prompt = document.getElementById('prompt').value;
    if (prompt) {
        document.getElementById("errorMsg").innerText = '';
        document.getElementById("error").style.display = 'none';
        document.getElementById("result").innerHTML = '';
        document.getElementById("loading").style.display = 'flex';
        document.getElementById("generateButton").disabled = true;
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt })
        })
        .then(response => {
            if (response.ok) {
                return response.blob();
            } else {
                return response.json().then(err => {
                    console.log(err.error);
                    document.getElementById("loading").style.display = 'none';
                    document.getElementById("errorMsg").innerText = err.error;
                    document.getElementById("error").style.display = 'flex';
                    throw new Error(err.error);
                });
            }
        }).then(blob => {
            document.getElementById("loading").style.display = 'none';
            const url = window.URL.createObjectURL(blob);
            document.getElementById("result").innerHTML = '<h2>Video generated successfully!</h2> <button><a href="' + url + '" download="video.mp4">Download Video</a></button>';
        });
        document.getElementById("generateButton").disabled = false;
    } else {
        alert('Please enter a prompt.');
    }
}