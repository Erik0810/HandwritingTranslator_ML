// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up file input label to show selected filename
    const fileInput = document.getElementById('imageUpload');
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            
            // Create a preview if it's an image
            const reader = new FileReader();
            reader.onload = function(e) {
                // Set the preview image source
                document.getElementById('imagePreview').src = e.target.result;
                
                // Set file info
                document.getElementById('imageName').textContent = file.name;
                document.getElementById('imageSize').textContent = formatFileSize(file.size);
                
                // Show preview, hide upload area
                document.getElementById('uploadArea').classList.add('hidden');
                document.getElementById('previewArea').classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        }
    });
    
    // New upload button click handler
    document.getElementById('newUploadBtn').addEventListener('click', function() {
        // Clear the file input
        document.getElementById('imageUpload').value = '';
        
        // Hide preview, show upload area
        document.getElementById('previewArea').classList.add('hidden');
        document.getElementById('uploadArea').classList.remove('hidden');
        
        // Clear result
        document.getElementById('result').innerHTML = '';
    });
    
    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' bytes';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }
    
    // Set up camera access
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('captureButton');
    const trainButton = document.getElementById('trainButton');
    
    // Access the camera
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                video.srcObject = stream;
            })
            .catch(function(error) {
                console.error("Camera error:", error);
                document.querySelector('.camera-section').innerHTML += 
                    `<div class="mt-4 p-3 bg-red-900/50 text-red-200 rounded-lg">
                        Camera access denied or not available
                    </div>`;
            });
    }
    
    // Capture image from camera
    captureButton.addEventListener('click', function() {
        // Add loading state
        this.disabled = true;
        this.innerHTML = `
            <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
        `;
        
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to base64 image
        const imageData = canvas.toDataURL('image/png');
        
        // Send to server
        fetch('/camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'image=' + encodeURIComponent(imageData)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Camera capture response:', data);
            if (data.success) {
                document.getElementById('result').innerHTML = `
                    <div class="flex items-start">
                        <div class="bg-blue-500/20 p-1 rounded mr-2">
                            <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                        </div>
                        <div>
                            <p class="font-medium text-white">Recognized Text:</p>
                            <p class="text-gray-300">${data.text}</p>
                        </div>
                    </div>
                `;
            } else {
                document.getElementById('result').innerHTML = `
                    <div class="flex items-start">
                        <div class="bg-red-500/20 p-1 rounded mr-2">
                            <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                        </div>
                        <div>
                            <p class="font-medium text-white">Error:</p>
                            <p class="text-red-300">${data.message}</p>
                        </div>
                    </div>
                `;
            }
            
            // Reset button
            captureButton.disabled = false;
            captureButton.textContent = 'Capture';
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerHTML = `
                <div class="flex items-start">
                    <div class="bg-red-500/20 p-1 rounded mr-2">
                        <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <div>
                        <p class="font-medium text-white">Error:</p>
                        <p class="text-red-300">${error}</p>
                    </div>
                </div>
            `;
            
            // Reset button
            captureButton.disabled = false;
            captureButton.textContent = 'Capture';
        });
    });
    
    // Set up WebSocket connection for real-time training updates if available
    let socket;
    try {
        socket = io();
        
        socket.on('connect', function() {
            console.log('WebSocket connected');
        });
        
        socket.on('training_update', function(data) {
            console.log('Training update:', data);
            
            // Update progress UI
            updateTrainingProgress(data);
        });
        
        socket.on('training_complete', function(data) {
            console.log('Training complete:', data);
            
            // Complete the progress bar
            document.getElementById('trainingPercentage').textContent = '100%';
            document.getElementById('trainingProgressBar').style.width = '100%';
            
            // Update the training status
            const trainingStatus = document.getElementById('trainingStatus');
            if (data.success) {
                trainingStatus.textContent = data.message;
                trainingStatus.className = 'text-green-400 text-sm';
                
                // Update model status
                const modelStatus = document.getElementById('modelStatus');
                modelStatus.textContent = 'Loaded';
                modelStatus.className = 'font-medium text-green-400';
                
                // Update the status indicator
                modelStatus.previousElementSibling.className = 'w-3 h-3 rounded-full mr-2 bg-green-500';
            } else {
                trainingStatus.textContent = 'Error: ' + data.message;
                trainingStatus.className = 'text-red-400 text-sm';
            }
            
            // Reset train button
            const trainButton = document.getElementById('trainButton');
            trainButton.disabled = false;
            trainButton.textContent = 'Train Model';
        });
        
        console.log('SocketIO initialized successfully');
    } catch (e) {
        console.warn('SocketIO not available:', e);
        // We'll use a fallback polling mechanism
    }
    
    function updateTrainingProgress(data) {
        // Update epoch info
        document.getElementById('trainingEpoch').textContent = `Epoch: ${data.epoch + 1}/${data.total_epochs}`;
        
        // Calculate overall progress
        const overallProgress = ((data.epoch * 100) + data.epoch_progress) / data.total_epochs;
        
        // Update progress bar
        document.getElementById('trainingPercentage').textContent = `${Math.round(overallProgress)}%`;
        document.getElementById('trainingProgressBar').style.width = `${overallProgress}%`;
        
        // Update metrics
        document.getElementById('trainingAccuracy').textContent = `${data.accuracy.toFixed(1)}%`;
        document.getElementById('trainingLoss').textContent = data.loss.toFixed(3);
        
        // Update time information
        const elapsedMinutes = Math.floor(data.time_elapsed / 60);
        const elapsedSeconds = Math.floor(data.time_elapsed % 60);
        document.getElementById('trainingTime').textContent = 
            `${elapsedMinutes.toString().padStart(2, '0')}:${elapsedSeconds.toString().padStart(2, '0')}`;
        
        const remainingMinutes = Math.floor(data.time_remaining / 60);
        const remainingSeconds = Math.floor(data.time_remaining % 60);
        document.getElementById('trainingETA').textContent = 
            `${remainingMinutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    // Train model button click handler
    trainButton.addEventListener('click', function() {
        const trainingStatus = document.getElementById('trainingStatus');
        const progressPanel = document.getElementById('trainingProgressPanel');
        
        // Add loading state
        this.disabled = true;
        this.innerHTML = `
            <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Training...
        `;
        
        trainingStatus.textContent = 'Training in progress...';
        trainingStatus.className = 'text-blue-400 text-sm animate-pulse';
        
        // Show progress panel and reset values
        progressPanel.classList.remove('hidden');
        document.getElementById('trainingEpoch').textContent = 'Epoch: 0/10';
        document.getElementById('trainingPercentage').textContent = '0%';
        document.getElementById('trainingProgressBar').style.width = '0%';
        document.getElementById('trainingAccuracy').textContent = '-';
        document.getElementById('trainingLoss').textContent = '-';
        document.getElementById('trainingTime').textContent = '00:00';
        document.getElementById('trainingETA').textContent = '--:--';
        
        // Start training
        fetch('/train', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Training response:', data);
            
            if (data.success) {
                trainingStatus.textContent = data.message;
                trainingStatus.className = 'text-green-400 text-sm';
                
                const modelStatus = document.getElementById('modelStatus');
                modelStatus.textContent = 'Loaded';
                modelStatus.className = 'font-medium text-green-400';
                
                // Update the status indicator
                modelStatus.previousElementSibling.className = 'w-3 h-3 rounded-full mr-2 bg-green-500';
            } else {
                trainingStatus.textContent = 'Error: ' + data.message;
                trainingStatus.className = 'text-red-400 text-sm';
                
                // Hide progress panel on error
                progressPanel.classList.add('hidden');
            }
            
            // Reset button
            trainButton.disabled = false;
            trainButton.textContent = 'Train Model';
        })
        .catch(error => {
            console.error('Error:', error);
            trainingStatus.textContent = 'Error: ' + error;
            trainingStatus.className = 'text-red-400 text-sm';
            
            // Hide progress panel on error
            progressPanel.classList.add('hidden');
            
            // Reset button
            trainButton.disabled = false;
            trainButton.textContent = 'Train Model';
        });
    });
});

function uploadFile() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
        return;
    }
    
    console.log('Uploading file:', file.name);
    
    // Add loading state to button
    const uploadButton = document.querySelector('button[onclick="uploadFile()"]');
    uploadButton.disabled = true;
    uploadButton.innerHTML = `
        <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Processing...
    `;
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Upload response:', data);
        if (data.success) {
            document.getElementById('result').innerHTML = `
                <div class="flex items-start">
                    <div class="bg-blue-500/20 p-1 rounded mr-2">
                        <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <div>
                        <p class="font-medium text-white">Recognized Text:</p>
                        <p class="text-gray-300">${data.text}</p>
                    </div>
                </div>
            `;
        } else {
            document.getElementById('result').innerHTML = `
                <div class="flex items-start">
                    <div class="bg-red-500/20 p-1 rounded mr-2">
                        <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                    </div>
                    <div>
                        <p class="font-medium text-white">Error:</p>
                        <p class="text-red-300">${data.message}</p>
                    </div>
                </div>
            `;
        }
        
        // Reset button
        uploadButton.disabled = false;
        uploadButton.textContent = 'Translate';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `
            <div class="flex items-start">
                <div class="bg-red-500/20 p-1 rounded mr-2">
                    <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                </div>
                <div>
                    <p class="font-medium text-white">Error:</p>
                    <p class="text-red-300">${error}</p>
                </div>
            </div>
        `;
        
        // Reset button
        uploadButton.disabled = false;
        uploadButton.textContent = 'Translate';
    });
} 