import { ScrollViewStyleReset } from 'expo-router/html';
import { type PropsWithChildren } from 'react';

export default function Root({ children }: PropsWithChildren) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <title>CloudMind Academy</title>
        <meta name="description" content="Learn full-stack development with CloudMind Academy" />

        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />

        <ScrollViewStyleReset />

        <style dangerouslySetInnerHTML={{ __html: globalStyles }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

const globalStyles = `
:root {
  --color-primary: #10B981;
  --color-primary-dark: #059669;
  --color-background: #FFFFFF;
  --color-text: #1F2937;
  --color-border: #E5E7EB;
}

body {
  background-color: var(--color-background);
  color: var(--color-text);
  font-family: 'Inter', sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root {
    --color-primary: #34D399;
    --color-primary-dark: #10B981;
    --color-background: #1F2937;
    --color-text: #F9FAFB;
    --color-border: #4B5563;
  }
}

.expo-scroll-view-style-reset {
  scrollbar-width: none;
  -ms-overflow-style: none;
}

.expo-scroll-view-style-reset::-webkit-scrollbar {
  display: none;
}

a {
  color: var(--color-primary);
  text-decoration: none;
}

a:hover {
  color: var(--color-primary-dark);
  text-decoration: underline;
}

button, .button {
  background-color: var(--color-primary);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  border: none;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

button:hover, .button:hover {
  background-color: var(--color-primary-dark);
}
`;