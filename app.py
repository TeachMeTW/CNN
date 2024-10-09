import streamlit as st
import streamlit.components.v1 as components

# Define the list of words/phrases you want to cycle through
words = [
    "Skibidi",
    "Toilet"
]

# HTML and CSS for the typewriter effect
typewriter_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        .typewriter {{
            font-family: 'Courier New', Courier, monospace;
            color: #FF6600; /* Nixie Orange */
            overflow: hidden; /* Ensures the text is not revealed until the animation */
            border-right: .15em solid #FF6600; /* The typewriter cursor in Nixie Orange */
            white-space: nowrap; /* Prevents the text from wrapping */
            margin: 0 auto; /* Centers the element */
            letter-spacing: .15em; /* Adjusts spacing between letters */
            font-size: 2em; /* Adjust font size as needed */
        }}
        /* Optional: Center the typewriter text */
        body {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }}
    </style>
</head>
<body>

    <div class="typewriter" id="typewriter"></div>

    <script>
        const words = {words};
        let currentWordIndex = 0;
        let currentCharIndex = 0;
        let isDeleting = false;
        const typingSpeed = 150; // milliseconds
        const deletingSpeed = 100;
        const pauseBetweenWords = 1500;

        const typewriterElement = document.getElementById('typewriter');

        function type() {{
            const currentWord = words[currentWordIndex];
            if (isDeleting) {{
                typewriterElement.innerHTML = currentWord.substring(0, currentCharIndex--);
                if (currentCharIndex < 0) {{
                    isDeleting = false;
                    currentWordIndex = (currentWordIndex + 1) % words.length;
                    setTimeout(type, 500);
                }} else {{
                    setTimeout(type, deletingSpeed);
                }}
            }} else {{
                typewriterElement.innerHTML = currentWord.substring(0, currentCharIndex++);
                if (currentCharIndex > currentWord.length) {{
                    isDeleting = true;
                    setTimeout(type, pauseBetweenWords);
                }} else {{
                    setTimeout(type, typingSpeed);
                }}
            }}
        }}

        // Start the typing effect
        type();
    </script>

</body>
</html>
"""

# Render the typewriter HTML in the Streamlit app
components.html(typewriter_html, height=50)

# Add a space between the typewriter and the image
st.markdown("---")
st.image(
    "https://i.imgur.com/PiVKciH.jpeg",
    caption="Skibidi Toilet",
    use_column_width=True
)
st.markdown("---")
video_file = open("skibidi.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)
