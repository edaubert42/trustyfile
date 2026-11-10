// TrustyFile — the page's script. Four small jobs, none of them required for
// the site to work: without JavaScript the upload form still posts, the page
// selector is simply absent of effect and the thumbs do nothing.
//
//   1. Upload: clicking or dropping on the zone sends the form.
//   2. Splash: one overlay while the server analyses the document.
//   3. Preview: the page selector swaps the image.
//   4. Feedback: thumbs up/down and the optional comment.
(function () {
    'use strict';

    // ---- Splash ------------------------------------------------------------
    // An analysis takes from one to a few tens of seconds, during which the
    // browser shows nothing new. The overlay says so. It is removed if the
    // visitor comes back to this page with the Back button (see pageshow).
    var splash = null;

    function raiseSplash(label) {
        if (splash) return;
        splash = document.createElement('div');
        splash.id = 'tf-splash';
        var logo = document.createElement('img');
        logo.src = '/logo.png';
        logo.alt = '';
        var spin = document.createElement('div');
        spin.className = 'tf-spin';
        splash.appendChild(logo);
        splash.appendChild(spin);
        if (label) {
            var caption = document.createElement('div');
            caption.className = 'tf-label';
            caption.textContent = label;
            splash.appendChild(caption);
        }
        document.body.appendChild(splash);
    }

    function lowerSplash() {
        if (!splash) return;
        splash.remove();
        splash = null;
    }

    window.addEventListener('pageshow', lowerSplash);
    var analysingLabel = document.body.dataset.analyzing || '';

    // Links that start an analysis (the sample document).
    document.querySelectorAll('a.tf-splash-link').forEach(function (link) {
        link.addEventListener('click', function () { raiseSplash(analysingLabel); });
    });

    // ---- Upload ------------------------------------------------------------
    var form = document.getElementById('tf-upload');
    if (form) {
        var input = form.querySelector('input[type=file]');
        var zone = document.getElementById('tf-dropzone');

        var send = function () {
            if (!input.files || input.files.length === 0) return;
            raiseSplash(analysingLabel);
            form.submit();
        };

        input.addEventListener('change', send);
        zone.addEventListener('click', function (event) {
            if (event.target !== input) input.click();
        });

        ['dragenter', 'dragover'].forEach(function (name) {
            zone.addEventListener(name, function (event) {
                event.preventDefault();
                zone.classList.add('tf-drag');
            });
        });
        ['dragleave', 'drop'].forEach(function (name) {
            zone.addEventListener(name, function (event) {
                event.preventDefault();
                zone.classList.remove('tf-drag');
            });
        });
        zone.addEventListener('drop', function (event) {
            if (!event.dataTransfer || event.dataTransfer.files.length === 0) return;
            input.files = event.dataTransfer.files;
            send();
        });
    }

    // ---- Preview page selector ---------------------------------------------
    var pageSelect = document.getElementById('tf-page-select');
    var previewImage = document.getElementById('tf-preview-img');
    if (pageSelect && previewImage) {
        pageSelect.addEventListener('change', function () {
            previewImage.src = pageSelect.dataset.src + pageSelect.value + '.png';
        });
    }

    // ---- Feedback ----------------------------------------------------------
    var feedback = document.getElementById('tf-feedback');
    if (feedback) {
        var thumbs = feedback.querySelectorAll('.tf-thumbs button');
        var commentForm = feedback.querySelector('.tf-feedback-form');
        var thanks = feedback.querySelector('.tf-feedback-thanks');
        var comment = feedback.querySelector('textarea');
        var vote = feedback.dataset.vote || null;      // "up", "down" or null
        var sent = feedback.dataset.sent === '1';      // comment already sent

        // Calls go out one after the other: the comment must reach the
        // server after the vote it belongs to. On the sample document the
        // server answers with a ticket (this visitor's vote, as the page is
        // shared by everybody); it is sent back with the following calls.
        var ticket = null;
        var queue = Promise.resolve();
        var post = function (body) {
            queue = queue.then(function () {
                if (ticket) body.ticket = ticket;
                return fetch(feedback.dataset.url, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body)
                });
            }).then(function (response) {
                return response.json();
            }).then(function (answer) {
                if (answer && answer.ticket) ticket = answer.ticket;
            }).catch(function () { /* feedback is best effort */ });
        };

        // The comment box follows a thumbs down; the thank-you line follows
        // a thumbs up, or a thumbs down once the comment is sent.
        var show = function () {
            thumbs.forEach(function (button) {
                button.setAttribute('aria-checked', button.dataset.vote === vote ? 'true' : 'false');
            });
            commentForm.hidden = !(vote === 'down' && !sent);
            thanks.hidden = !(vote === 'up' || (vote === 'down' && sent));
        };

        thumbs.forEach(function (button) {
            button.addEventListener('click', function () {
                if (vote === button.dataset.vote) {
                    // Same thumb again: the selection clears, the stored
                    // vote stays (one opinion, one row).
                    vote = null;
                } else {
                    vote = button.dataset.vote;
                    post({vote: vote});
                }
                show();
            });
        });

        commentForm.addEventListener('submit', function (event) {
            event.preventDefault();
            post({comment: comment.value});
            sent = true;
            show();
        });

        show();
    }
})();
