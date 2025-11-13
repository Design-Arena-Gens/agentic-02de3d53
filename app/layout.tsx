export const metadata = {
  title: 'STEMI Detection AI - ECG Analysis',
  description: 'ML-powered STEMI detection from ECG images',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0 }}>{children}</body>
    </html>
  )
}
