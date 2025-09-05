import React from "react";

// Simple notification component
export const Notification = ({ message, type = "info", onClose }) => {
  const typeColors = {
    success: "background: #d4edda; color: #155724; border-color: #c3e6cb;",
    error: "background: #f8d7da; color: #721c24; border-color: #f5c6cb;",
    warning: "background: #fff3cd; color: #856404; border-color: #ffeaa7;",
    info: "background: #d1ecf1; color: #0c5460; border-color: #bee5eb;",
  };

  return (
    <div
      style={{
        position: "fixed",
        top: "20px",
        right: "20px",
        padding: "1rem",
        borderRadius: "0.375rem",
        border: "1px solid",
        zIndex: 1000,
        ...typeColors[type],
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        boxShadow: "0 0.25rem 0.5rem rgba(0,0,0,0.1)",
      }}
    >
      <span>{message}</span>
      <button
        onClick={onClose}
        style={{
          background: "none",
          border: "none",
          fontSize: "1.25rem",
          cursor: "pointer",
          padding: "0",
          marginLeft: "0.5rem",
        }}
      >
        ×
      </button>
    </div>
  );
};

// Notification container to manage multiple notifications
export const NotificationContainer = ({ notifications, onRemove }) => {
  return (
    <div>
      {notifications.map((notification) => (
        <Notification
          key={notification.id}
          message={notification.message}
          type={notification.type}
          onClose={() => onRemove(notification.id)}
        />
      ))}
    </div>
  );
};
