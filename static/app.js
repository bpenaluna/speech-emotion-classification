const form = document.getElementById('upload-form');
const result = document.getElementById('result');

form.addEventListener('submit', async (e) => {

    e.preventDefault(); // prevent normal form submit
    result.textContent = "Predicting..."
    
    const formData = new FormData(form);
    
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const text = await response.text();

    result.textContent = text;
});