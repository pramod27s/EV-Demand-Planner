export default function ErrorBanner({ message }: { message: string }) {
  const pipeline = message.includes("Run the ML pipeline");
  return (
    <div role="alert" className="rounded border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-800">
      <div className="font-semibold">{pipeline ? "Results not available yet" : "Error"}</div>
      <div className="mt-0.5">{message}</div>
    </div>
  );
}
