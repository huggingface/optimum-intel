<div id="copyMsg" style="display:none; position:fixed; top:10px; left:50%; transform:translateX(-50%); background:green; color:white; padding:5px 10px; border-radius:4px; z-index:1000; font-family:sans-serif; font-size:14px;">
    Command copied to clipboard
</div>
<script>
    function copyCommand(command) {
        navigator.clipboard.writeText(command);
        let m = document.getElementById('copyMsg');
        m.style.display = 'block';
        clearTimeout(window._copyTimeout);
        window._copyTimeout = setTimeout(() => m.style.display = 'none', 1000);
    }
</script>