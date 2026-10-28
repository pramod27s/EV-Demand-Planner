import { Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/ui/Navbar";
import Workspace from "./pages/Workspace";
import LabLayout from "./pages/lab/LabLayout";
import Method from "./pages/lab/Method";
import Optimization from "./pages/lab/Optimization";
import Performance from "./pages/lab/Performance";

export default function App() {
  return (
    <div className="flex h-screen flex-col">
      <Navbar />
      <div className="min-h-0 flex-1">
        <Routes>
          {/* the tool itself: full-screen map workspace */}
          <Route path="/" element={<Workspace />} />
          {/* technical section */}
          <Route path="/lab" element={<LabLayout />}>
            <Route index element={<Performance />} />
            <Route path="optimization" element={<Optimization />} />
            <Route path="method" element={<Method />} />
          </Route>
          {/* old URLs */}
          <Route path="/analyze" element={<Navigate to="/" replace />} />
          <Route path="/predict" element={<Navigate to="/" replace />} />
          <Route path="/stations" element={<Navigate to="/" replace />} />
          <Route path="/optimization" element={<Navigate to="/lab/optimization" replace />} />
          <Route path="/about" element={<Navigate to="/lab/method" replace />} />
          <Route path="*" element={<p className="p-6 text-zinc-600">Page not found.</p>} />
        </Routes>
      </div>
    </div>
  );
}
