importScripts('./ai.js');

self.addEventListener('message', function(e) {
    console.log(e.data);
    _initialize_ai();
    self.postMessage(0);
}, false);
